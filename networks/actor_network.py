import jax
import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax.random as jrng

EPS = 1e-6


def embed(
        obs, # [84, 84, 4]
        config,
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
        muzero_params,
        obs,
        actions,
        config,
):
    (embed_params, reward_params, value_params,
     policy_params, dynamics_params) = muzero_params
    s = embed_t.apply(embed_params, obs, config)
    def f(state, action):
        r = reward_t.apply(reward_params, state, config)
        v = value_t.apply(value_params, state, config)
        pi = policy_t.apply(policy_params, state, config)
        next_state = dynamics_t.apply(dynamics_params, state, action, config)
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