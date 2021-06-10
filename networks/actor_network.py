from typing import NamedTuple, Callable, Mapping, Union
import common

import jax
import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax.random as jrng

EPS = 1e-6


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


def reward(
        state,
        config
):
    return hk.Sequential([
        hk.Linear(config['embedding_size']),
        jax.nn.relu,
        hk.Linear(1)
    ])(state)[0]


def value(
        state,
        config
):
    return hk.Sequential([
        hk.Linear(config['embedding_size']),
        jax.nn.relu,
        hk.Linear(1)
    ])(state)[0]


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


def rollout_model(
        muzero_agent: 'MuZeroAgent',
        obs,
        actions,
        config,
):
    params = muzero_agent.params
    s = muzero_agent.embed.apply(params.embed, obs, config)
    def f(state, action):
        r = muzero_agent.reward.apply(params.reward, state, config)
        v = muzero_agent.value.apply(params.value, state, config)
        pi = muzero_agent.policy.apply(params.policy, state, config)
        next_state = muzero_agent.dynamics.apply(params.dynamics, state, action, config)
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
    chex.assert_shape(env_pi_rollout, (n + K, A))
    chex.assert_shape(model_pi_rollout, (K, A))
    env_pi_rollout = env_pi_rollout[:K, :]
    cross_entropies = -jnp.sum(env_pi_rollout * jnp.log(model_pi_rollout + EPS), axis=1)
    return jnp.sum(cross_entropies, axis=0)


class MuZeroParams(NamedTuple):
    embed: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    policy: jnp.ndarray
    dynamics: jnp.ndarray


class MuZeroComponents(NamedTuple):
    embed: Callable[[jnp.ndarray, common.Config], jnp.ndarray]
    reward: Callable[[jnp.ndarray, common.Config], jnp.ndarray]
    value: Callable[[jnp.ndarray, common.Config], jnp.ndarray]
    policy: Callable[[jnp.ndarray, common.Config], jnp.ndarray]
    dynamics: Callable[[jnp.ndarray, jnp.ndarray, common.Config], jnp.ndarray]


class MuZeroAgent:

    def __init__(
            self,
            key: jrng.PRNGKey,
            components: MuZeroComponents,
            config: common.Config,
    ):
        dummy_obs = np.zeros(config['obs_shape'], dtype=np.float32)
        dummy_state = np.zeros(config['embedding_size'], dtype=np.float32)
        dummy_action = 0

        self.reward = hk.transform(components.reward)
        self.embed = hk.transform(components.embed)
        self.value = hk.transform(components.value)
        self.policy = hk.transform(components.policy)
        self.dynamics = hk.transform(components.dynamics)

        key, *param_keys = jrng.split(key, 6)

        self.params = MuZeroParams(
            reward=self.reward.init(param_keys[0], dummy_state, config),
            embed=self.embed.init(param_keys[1], dummy_obs, config),
            value=self.value.init(param_keys[2], dummy_state, config),
            policy=self.policy.init(param_keys[3], dummy_state, config),
            dynamics=self.dynamics.init(param_keys[4], dummy_state, dummy_action, config)
        )