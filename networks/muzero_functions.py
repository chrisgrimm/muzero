from typing import NamedTuple, Callable, Mapping, Union, Tuple
import common

import jax
import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax.random as jrng
import optax
from jax.tree_util import tree_map
from networks import mcts
from networks.muzero_def import MuZeroParams, MuZeroComponents
from networks.residual_block import ResidualBlock


def _stack_obs_and_actions(
        img_obs: jnp.ndarray,
        action_obs: jnp.ndarray,
        config: common.Config
):
    img_obs = jnp.transpose(img_obs, [1, 2, 0, 3]) / 255.  # [96, 96, num_back, 3]
    action_obs = (action_obs / config['num_actions'])[:, :, None, :]
    action_obs = jnp.ones(img_obs.shape[:3] + (1,), dtype=jnp.float32) * action_obs
    stacked = jnp.concatenate([img_obs, action_obs], axis=3)  # [96, 96, num_back, 4]
    h, w, num_back, channels = jnp.shape(stacked)
    stacked = jnp.reshape(stacked, [h, w, num_back * channels])  # [96, 96, num_back * 4]
    return stacked


def embed(
        obs: Tuple[jnp.ndarray, jnp.ndarray], # ([num_back, 96, 96, 3], [num_back])
        config: common.Config,
):
    stacked = _stack_obs_and_actions(obs[0], obs[1], config)
    return hk.Sequential([
        hk.Conv2D(128, 3, 2, padding='VALID'),
        jax.nn.relu,
        ResidualBlock(128, 3),
        ResidualBlock(128, 3),
        hk.Conv2D(256, 3, 2, padding='VALID'),
        jax.nn.relu,
        ResidualBlock(256, 3),
        ResidualBlock(256, 3),
        ResidualBlock(256, 3),
        hk.AvgPool(2, 2, 'VALID'),
        ResidualBlock(256, 3),
        ResidualBlock(256, 3),
        ResidualBlock(256, 3),
        hk.AvgPool(2, 2, 'VALID'),
    ])(stacked)


embed_t = hk.transform(embed)


def dynamics(
        state,
        action,
        config
):
    action = jnp.broadcast_to(jax.nn.one_hot(action, config['num_actions']), state.shape)
    sa = jnp.concatenate([state, action], axis=2)
    return hk.Sequential([
        hk.Conv2D(256, 3, 1, padding='VALID'),
        jax.nn.relu,
        ResidualBlock(256, 3),
        ResidualBlock(256, 3)
    ])(sa)


dynamics_t = hk.transform(dynamics)


def reward(
        state,
        config
):
    return hk.Sequential([
        hk.Conv2D(64, 3, 1, padding='VALID'),
        jax.nn.relu,
        hk.Conv2D(32, 3, 1, padding='VALID'),
        jax.nn.relu,
        lambda x: jnp.reshape(x, [32 * 6 * 6]),
        hk.Linear(1)
    ])(state)[0]


reward_t = hk.transform(reward)


def value(
        state,
        config
):
    return hk.Sequential([
        hk.Conv2D(64, 3, 1, padding='VALID'),
        jax.nn.relu,
        hk.Conv2D(32, 3, 1, padding='VALID'),
        jax.nn.relu,
        lambda x: jnp.reshape(x, [32 * 6 * 6]),
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
        # TODO this indexing looks wrong.
        env_r = env_r_rollout[k+1:k+n]
        gamma_terms = np.array([gamma ** t for t in range(n-1)])
        env_n_step = jnp.sum(env_r * gamma_terms, axis=0)
        target = jax.lax.stop_gradient(env_n_step + gamma**n * v_bootstraps[k])
        loss_term = (model_v_rollout[k] - target)**2
        return loss + loss_term, target

    v_loss, targets = jax.lax.scan(f, 0.0, np.arange(K))

    return v_loss, targets


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


def process_trajectories(
        obs_traj: jnp.ndarray,  # [back + K + n, ...]
        a_traj: jnp.ndarray,  # [back + K + n, ...]
        r_traj: jnp.ndarray,  # [back + K + n, ...]
        search_pi_traj: jnp.ndarray,  # ...
        search_v_traj: jnp.ndarray,  # ...
        config: common.Config,
):
    # [back + K + n, 96, 96, 3] --> [K + n, back, 96, 96, 3]
    num_hist = config['num_hist']
    K, n = config['model_rollout_length'], config['env_rollout_length']
    f = lambda t: (obs_traj[t:num_hist+t], a_traj[t:num_hist+t])
    obs_traj = jax.vmap(f)(jnp.arange(0, K + n))  # ([K + n, back, 96, 96, 3], ...)
    a_traj = a_traj[num_hist:]
    r_traj = r_traj[num_hist:]
    # TODO search_pi_traj is going to be weird. figure this out tomorrow.
    search_pi_traj = search_pi_traj[num_hist:]
    search_v_traj = search_v_traj[num_hist:]
    return obs_traj, a_traj, r_traj, search_pi_traj, search_v_traj





def muzero_loss(
        muzero_params: MuZeroParams,
        muzero_comps: MuZeroComponents,
        obs_traj: jnp.ndarray,  # [back + K + n, *obs_size]
        a_traj: jnp.ndarray,  # [back + K + n]
        r_traj: jnp.ndarray,  # [back + K + n]
        search_pi_traj: jnp.ndarray,  # [back + K + n]
        search_v_traj: jnp.ndarray,  # [back + K + n]
        importance_weight: jnp.ndarray,  # []
        config: common.Config,
):
    obs_traj, a_traj, r_traj, search_pi_traj, search_v_traj = process_trajectories(
        obs_traj, a_traj, r_traj, search_pi_traj, search_v_traj, config
    )
    n, K = config['environment_rollout_length'], config['model_rollout_length']
    # each is [K, ...]
    first_obs = tree_map(lambda x: x[0], obs_traj)
    model_r_traj, model_v_traj, model_pi_traj = rollout_model(
        muzero_params, muzero_comps, first_obs, a_traj[:K], config)
    r_loss_term = r_loss(r_traj, model_r_traj, config)

    #v_mcts = jax.vmap(mcts.run_and_get_value, (None, None, 0, None), 0)(
    #    muzero_params, muzero_comps, key, obs_traj[n:], config)
    v_loss_term, targets = v_loss(r_traj, search_v_traj, model_v_traj, config)
    new_priority = jnp.abs(targets[0] - search_v_traj[0])

    #pi_traj = jax.vmap(mcts.run_and_get_policy, (None, None, 0, None, None), 0)(
    #    muzero_params, muzero_comps, key, obs_traj[:K], jnp.array(1.0), config)
    pi_loss_term = policy_loss(search_pi_traj[:K], model_pi_traj, config)

    loss = r_loss_term + v_loss_term + pi_loss_term
    loss = importance_weight * loss
    return loss, new_priority


def train_muzero(
        muzero_params: MuZeroParams,
        muzero_comps: MuZeroComponents,
        opt_state,
        optimizer,
        obs_traj: jnp.ndarray,  # [B, num_back + K + n, *obs_size]
        a_traj: jnp.ndarray,  # [B, num_back + K + n]
        r_traj: jnp.ndarray,  # [B, num_back + K + n]
        search_pi_traj: jnp.ndarray,  # [B, num_back + K + n, num_actions]
        search_v_traj: jnp.ndarray,  # [B, num_back + K + n, v]
        importance_weights: jnp.ndarray,  # [B,]
        config: common.Config,
):
    batched_loss = jax.vmap(muzero_loss, (None, None, 0, 0, 0, 0, 0, 0, None), (0, 0))

    def batched_loss(*args, batched_loss=batched_loss):
        loss, priorities = batched_loss(*args)
        return jnp.mean(loss, axis=0), priorities

    (loss, priorities), grads = jax.value_and_grad(batched_loss, has_aux=True)(
        muzero_params, muzero_comps,
        obs_traj, a_traj, r_traj, search_pi_traj, search_v_traj, importance_weights, config)
    updates, opt_state = optimizer.update(grads, opt_state, muzero_params)
    muzero_params = optax.apply_updates(muzero_params, updates)
    return loss, priorities, opt_state, muzero_params