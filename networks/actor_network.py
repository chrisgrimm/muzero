from typing import NamedTuple, Callable, Mapping, Union, Tuple
import common

import jax
import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax.random as jrng
import optax

from networks import mcts
from networks.muzero_def import MuZeroParams, MuZeroComponents


def embed(
        obs, # [84, 84, 4]
        config: common.Config,
):
    return hk.Sequential([
        hk.Conv2D(16, 8, 4, padding='VALID'),
        jax.nn.relu,
        hk.Conv2D(32, 4, 2, padding='VALID'),
        jax.nn.relu,
        hk.Conv2D(32, 3, 1, padding='VALID'),
        jax.nn.relu,
        hk.Flatten(),
        hk.Linear(config['embedding_size'])
    ])(obs)


embed_t = hk.transform(embed)


def dynamics(
        state,
        action,
        config
):
    action = jax.nn.one_hot(action, config['num_actions'])
    sa = jnp.concatenate([state, action], axis=0)
    return hk.Sequential([
        hk.Linear(config['embedding_size']),
        jax.nn.relu,
    ] * config['depth'])(sa)


dynamics_t = hk.transform(dynamics)


def reward(
        state,
        config
):
    return hk.Sequential([
        hk.Linear(config['embedding_size']),
        jax.nn.relu,
        hk.Linear(1)
    ])(state)[0]


reward_t = hk.transform(reward)


def value(
        state,
        config
):
    return hk.Sequential([
        hk.Linear(config['embedding_size']),
        jax.nn.relu,
        hk.Linear(1)
    ])(state)[0]


value_t = hk.transform(value)


def policy(
        state,
        config
):
    return hk.Sequential([
        hk.Linear(config['embedding_size']),
        jax.nn.relu,
        hk.Linear(config['num_actions']),
        jax.nn.softmax,
    ])(state)


policy_t = hk.transform(policy)


def rollout_model(
        muzero_params: MuZeroParams,
        muzero_comps: MuZeroComponents,
        obs,
        actions,
        config,
):
    s = muzero_comps.embed.apply(muzero_params.embed, obs, config)
    def f(state, action):
        r = muzero_comps.reward.apply(muzero_params.reward, state, config)
        v = muzero_comps.value.apply(muzero_params.value, state, config)
        pi = muzero_comps.policy.apply(muzero_params.policy, state, config)
        next_state = muzero_comps.dynamics.apply(muzero_params.dynamics, state, action, config)
        return next_state, (r, v, pi)

    _, (r_traj, v_traj, pi_traj) = jax.lax.scan(f, s, actions)
    return r_traj, v_traj, pi_traj


def r_loss(
        env_r_rollout, # [K + n]
        model_r_rollout, # [K]
        config
):
    chex.assert_shape(model_r_rollout, (config['model_rollout_length'],))
    chex.assert_shape(env_r_rollout, (config['model_rollout_length'] + config['environment_rollout_length'],))
    env_r_rollout = env_r_rollout[:config['model_rollout_length']]
    return jnp.sum((env_r_rollout - model_r_rollout)**2, axis=0)


def v_loss(
        env_r_rollout, # [K + n]
        v_bootstraps, # [K]
        model_v_rollout, # [K]
        config
):
    K = config['model_rollout_length']
    n = config['environment_rollout_length']
    gamma = config['gamma']
    chex.assert_shape(env_r_rollout, (config['model_rollout_length'] + config['environment_rollout_length'],))
    chex.assert_shape(v_bootstraps, (config['model_rollout_length'],))
    chex.assert_shape(model_v_rollout, (config['model_rollout_length'],))

    def f(loss, k):
        env_r = env_r_rollout[k+1:k+n]
        gamma_terms = np.array([gamma ** t for t in range(n-1)])
        env_n_step = jnp.sum(env_r * gamma_terms, axis=0)
        target = jax.lax.stop_gradient(env_n_step + gamma**n * v_bootstraps[k])
        loss_term = (model_v_rollout[k] - target)**2
        return loss + loss_term, None

    v_loss, _ = jax.lax.scan(f, 0.0, np.arange(K))

    return v_loss


def policy_loss(
        env_pi_rollout,
        model_pi_rollout,
        config
):
    K = config['model_rollout_length']
    n = config['env_rollout_length']
    A = config['num_actions']
    chex.assert_shape(env_pi_rollout, (K, A))
    chex.assert_shape(model_pi_rollout, (K, A))
    cross_entropies = -jnp.sum(env_pi_rollout * jnp.log(model_pi_rollout + common.EPS), axis=1)
    return jnp.sum(cross_entropies, axis=0)


def muzero_loss(
        muzero_params: MuZeroParams,
        muzero_comps: MuZeroComponents,
        key: jrng.PRNGKey,
        obs_traj: jnp.ndarray, # [K + n, *obs_size]
        a_traj: jnp.ndarray, # [K + n]
        r_traj: jnp.ndarray, # [K + n]
        config: common.Config,
):
    n, K = config['environment_rollout_length'], config['model_rollout_length']
    # each is [K, ...]
    model_r_traj, model_v_traj, model_pi_traj = rollout_model(
        muzero_params, muzero_comps, obs_traj[0], a_traj[:K], config)
    r_loss_term = r_loss(r_traj, model_r_traj, config)

    v_mcts = jax.vmap(mcts.run_and_get_value, (None, None, 0, None), 0)(
        muzero_params, muzero_comps, key, obs_traj[n:], config)
    v_loss_term = v_loss(r_traj, v_mcts, model_v_traj, config)

    pi_traj = jax.vmap(mcts.run_and_get_policy, (None, None, 0, None, None), 0)(
        muzero_params, muzero_comps, key, obs_traj[:K], jnp.array(1.0), config)
    pi_loss_term = policy_loss(pi_traj, model_pi_traj, config)

    loss = r_loss_term + v_loss_term + pi_loss_term
    return loss


def train_muzero(
        muzero_params: MuZeroParams,
        muzero_comps: MuZeroComponents,
        opt_state,
        optimizer,
        key: jrng.PRNGKey,
        obs_traj: jnp.ndarray,  # [B, K + n, *obs_size]
        a_traj: jnp.ndarray,  # [B, K + n]
        r_traj: jnp.ndarray,  # [B, K + n]
        config: common.Config,
):
    batched_loss = jax.vmap(muzero_loss, (None, None, None, 0, 0, 0, None))
    batched_loss = lambda *args: jnp.mean(batched_loss(*args), axis=0)
    loss, grads = jax.value_and_grad(batched_loss)(
        muzero_params, muzero_comps, key, obs_traj, a_traj, r_traj, config)
    updates, opt_state = optimizer.update(grads, opt_state, muzero_params)
    muzero_params = optax.apply_updates(muzero_params, updates)
    return loss, opt_state, muzero_params