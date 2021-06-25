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
    action_obs = (action_obs / config['num_actions'])[None, None, :, None]
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
    x = hk.Conv2D(128, 3, 2, padding='SAME')(stacked)
    x = jax.nn.relu(x)
    x = ResidualBlock(128, 3)(x)
    x = ResidualBlock(128, 3)(x)
    x = hk.Conv2D(256, 3, 2, padding='SAME')(x)
    x = jax.nn.relu(x)
    x = ResidualBlock(256, 3)(x)
    x = ResidualBlock(256, 3)(x)
    x = ResidualBlock(256, 3)(x)
    x = hk.AvgPool((2, 2, 1), (2, 2, 1), 'SAME')(x)
    x = ResidualBlock(256, 3)(x)
    x = ResidualBlock(256, 3)(x)
    x = ResidualBlock(256, 3)(x)
    x = hk.AvgPool((2, 2, 1), (2, 2, 1), 'SAME')(x)
    return x


embed_t = hk.transform(embed)


def dynamics(
        state,
        action,
        config
):
    state = ((state - jnp.min(state, keepdims=True)) /
             (jnp.max(state, keepdims=True) - jnp.min(state, keepdims=True) + common.EPS))
    action = jax.nn.one_hot(action, config['num_actions'])[None, None, :]
    action = jnp.tile(action, (*state.shape[:2], 1))
    sa = jnp.concatenate([state, action], axis=2)
    return hk.Sequential([
        hk.Conv2D(256, 3, 1, padding='SAME'),
        jax.nn.relu,
        ResidualBlock(256, 3),
        ResidualBlock(256, 3)
    ])(sa)


dynamics_t = hk.transform(dynamics)


def get_categorical(
        x,  # []
        config: common.Config,
):
    delta_z = (config['cat_max'] - config['cat_min']) / (config['num_cat'] - 1)
    bins = jnp.array([config['cat_min'] + i * delta_z for i in range(config['num_cat'])])
    overlaps = jnp.abs(x[None] - bins)  # [num_bins]
    return overlaps


def get_scalar(
        c,
        config: common.Config,
):
    delta_z = (config['cat_max'] - config['cat_min']) / (config['num_cat'] - 1)
    bins = jnp.array([config['cat_min'] + i * delta_z for i in range(config['num_cat'])])
    return jnp.sum(bins * c, axis=0)


def target_transform(t):
    eps = 0.001
    return jnp.sign(t) * (jnp.sqrt(jnp.abs(t) + 1) - 1) + t * eps


def invert_target_transform(h):
    eps = 0.001
    inner = (jnp.sqrt(1 + 4*eps*(jnp.abs(h) + 1 + eps)) - 1) / (2*eps)
    return jnp.sign(h) * (inner**2 - 1)


def reward(
        state,
        config
):
    return hk.Sequential([
        hk.Conv2D(64, 3, 1, padding='SAME'),
        jax.nn.relu,
        hk.Conv2D(32, 3, 1, padding='SAME'),
        jax.nn.relu,
        lambda x: jnp.reshape(x, [-1]),
        hk.Linear(config['num_cat']),
        jax.nn.softmax
    ])(state)


reward_t = hk.transform(reward)


def value(
        state,
        config
):
    return hk.Sequential([
        hk.Conv2D(64, 3, 1, padding='SAME'),
        jax.nn.relu,
        hk.Conv2D(32, 3, 1, padding='SAME'),
        jax.nn.relu,
        lambda x: jnp.reshape(x, [-1]),
        hk.Linear(config['num_cat']),
        jax.nn.softmax
    ])(state)


value_t = hk.transform(value)


def policy(
        state,
        config
):
    return hk.Sequential([
        hk.Conv2D(64, 3, 1, padding='SAME'),
        jax.nn.relu,
        hk.Conv2D(32, 3, 1, padding='SAME'),
        jax.nn.relu,
        lambda x: jnp.reshape(x, [-1]),
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
    s = muzero_comps.embed.apply(muzero_params.embed, None, obs, config)
    def f(state, action):
        r = muzero_comps.reward.apply(muzero_params.reward, None, state, config)
        v = muzero_comps.value.apply(muzero_params.value, None, state, config)
        pi = muzero_comps.policy.apply(muzero_params.policy, None, state, config)
        next_state = muzero_comps.dynamics.apply(muzero_params.dynamics, None, state, action, config)
        return next_state, (r, v, pi)
    final_s, (r_traj, v_traj, pi_traj) = jax.lax.scan(f, s, actions)
    final_r, final_v, final_pi = (
        muzero_comps.reward.apply(muzero_params.reward, None, final_s, config),
        muzero_comps.value.apply(muzero_params.value, None, final_s, config),
        muzero_comps.policy.apply(muzero_params.policy, None, final_s, config))
    r_traj = jnp.concatenate([r_traj, final_r[None, ...]], axis=0)
    v_traj = jnp.concatenate([v_traj, final_v[None, ...]], axis=0)
    pi_traj = jnp.concatenate([pi_traj, final_pi[None, ...]], axis=0)
    return r_traj, v_traj, pi_traj


def r_loss(
        env_r_rollout, # [K + n]
        model_r_rollout, # [K, num_bins]
        config
):
    env_r_rollout = env_r_rollout[:config['model_rollout_length']+1]
    cat_env_r_rollout = jax.vmap(get_categorical, (0, None), 0)(env_r_rollout, config)  # [K, num_bins]
    cross_ents = jax.vmap(_cross_entropy, (0, 0), 0)(cat_env_r_rollout, model_r_rollout)
    return jnp.sum(cross_ents, axis=0)


def _cross_entropy(p, q):
    return -jnp.sum(p * jnp.log(q + common.EPS), axis=0)


def v_loss(
        env_r_rollout, # [K + n]
        v_bootstraps, # [K]
        model_v_rollout, # [K]
        config
):
    K = config['model_rollout_length']
    n = config['env_rollout_length']
    gamma = config['gamma']
    # chex.assert_shape(env_r_rollout, (config['model_rollout_length'] + config['env_rollout_length'],))
    # chex.assert_shape(v_bootstraps, (config['model_rollout_length'],))
    # chex.assert_shape(model_v_rollout, (config['model_rollout_length'],))

    r_stack = np.stack([np.arange(k+1, k+n+1) for k in range(0, K+1)])  # [K+1, n]
    env_r = env_r_rollout[r_stack]  # [K+1, n]
    v = v_bootstraps[np.arange(n, K+1+n)]  # [K+1]

    gamma_terms = np.array([gamma ** t for t in range(n)])  # [n]
    env_n_step = jnp.sum(env_r * gamma_terms[None, :], axis=1)  # [K+1]
    unscaled_target = jax.lax.stop_gradient(env_n_step + gamma**n * v)
    target = jax.vmap(target_transform)(unscaled_target)
    categorical_target = jax.vmap(get_categorical, (0, None), 0)(target, config)
    loss_term = jax.vmap(_cross_entropy, (0, 0), 0)(categorical_target, model_v_rollout)  # [K+1]
    return jnp.sum(loss_term, axis=0), unscaled_target



def policy_loss(
        env_pi_rollout,
        model_pi_rollout,
        config
):
    cross_entropies = jax.vmap(_cross_entropy, (0, 0), 0)(env_pi_rollout, model_pi_rollout)
    return jnp.sum(cross_entropies, axis=0)


def process_trajectories(
        obs_traj: jnp.ndarray,  # [back + K + n, ...]
        a_traj: jnp.ndarray,  # [back + K + n, ...]
        r_traj: jnp.ndarray,  # [back + K + n, ...]
        search_pi_traj: jnp.ndarray,  # ...
        search_v_traj: jnp.ndarray,  # ...
        config: common.Config,
):
    num_stack = config['num_stack']
    first_obs = (obs_traj[:num_stack, :, :, :], a_traj[:num_stack])
    a_traj = a_traj[num_stack-1:]
    r_traj = r_traj[num_stack-1:]
    # TODO search_pi_traj is going to be weird. figure this out tomorrow.
    search_pi_traj = search_pi_traj[num_stack-1:]
    search_v_traj = search_v_traj[num_stack-1:]
    return first_obs, a_traj, r_traj, search_pi_traj, search_v_traj


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
    first_obs, a_traj, r_traj, search_pi_traj, search_v_traj = process_trajectories(
        obs_traj, a_traj, r_traj, search_pi_traj, search_v_traj, config
    )
    n, K = config['env_rollout_length'], config['model_rollout_length']
    # each is [K, ...]
    model_r_traj, model_v_traj, model_pi_traj = rollout_model(
        muzero_params, muzero_comps, first_obs, a_traj[1:K+1], config)
    # K+1 terms for each

    r_loss_term = r_loss(r_traj, model_r_traj, config)

    #v_mcts = jax.vmap(mcts.run_and_get_value, (None, None, 0, None), 0)(
    #    muzero_params, muzero_comps, key, obs_traj[n:], config)
    v_loss_term, targets = v_loss(r_traj, search_v_traj, model_v_traj, config)
    new_priority = jnp.abs(targets[0] - search_v_traj[0])

    #pi_traj = jax.vmap(mcts.run_and_get_policy, (None, None, 0, None, None), 0)(
    #    muzero_params, muzero_comps, key, obs_traj[:K], jnp.array(1.0), config)
    pi_loss_term = policy_loss(search_pi_traj[:K+1], model_pi_traj, config)

    loss = (1 / K) * (r_loss_term + v_loss_term + pi_loss_term)
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