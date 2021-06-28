import jax
import numpy as np
import ray
import jax.random as jrng
import jax.numpy as jnp
import common

import environments.history_buffer
import main
from networks import mcts, muzero_def, muzero_functions
from jax.config import config as jax_config

jax_config.update('jax_disable_jit', True)

P_a1 = jnp.array([[0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1]], dtype=np.float32)
P_a2 = jnp.array([[0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1]], dtype=np.float32)

R = jnp.array([0, 1, 0, 3], dtype=np.float32)


def dummy_embed(x: jnp.ndarray, config: common.Config):
    return jax.nn.one_hot(x, 4)


def dummy_reward(x: jnp.ndarray, config: common.Config):
    return R[jnp.argmax(x)]


def dummy_value(x: jnp.ndarray, config: common.Config):
    return jnp.array(0, dtype=jnp.float32)


def dummy_policy(x: jnp.ndarray, config: common.Config):
    return jnp.array([0.5, 0.5], dtype=jnp.float32)


def dummy_dynamics(x: jnp.ndarray, a: jnp.ndarray, config: common.Config):
    s_idx = jnp.argmax(x)
    next_s_idx = jax.lax.cond(a == 0,
                        lambda _: jnp.argmax(P_a1[s_idx, :]),
                        lambda _: jnp.argmax(P_a2[s_idx, :]),
                        operand=None)
    return jax.nn.one_hot(next_s_idx, 4)

def build_muzero(config):

    muzero_params, muzero_comps = muzero_def.init_muzero(
        key=jrng.PRNGKey(1234),
        dummy_obs=jnp.array(0, dtype=jnp.uint8),
        dummy_action=jnp.array(0, dtype=jnp.uint8),
        embed=dummy_embed,
        reward=dummy_reward,
        value=dummy_value,
        policy=dummy_policy,
        dynamics=dummy_dynamics,
        config=config,
        process_reward=lambda x: x,
        process_value=lambda x: x,
    )
    return muzero_params, muzero_comps


def mcts_testing(obs, muzero_params, muzero_comps, config):
    action, policy, value = mcts.run_and_get_actor_quantities(
        muzero_params,
        muzero_comps,
        jrng.PRNGKey(1234),
        obs,
        np.array(1.0),
        config,
    )



if __name__ == '__main__':
    config = {
        'gamma': 0.5,
        'num_stack': 32,
        'obs_shape': (),
        'embedding_shape': (4,),
        'num_actions': 2,  # ?
        'num_simulations': 500,
        'model_rollout_length': 5,
        'env_rollout_length': 10,
        'update_actor_params_every': 1000,
        'update_temperature_every': 1000,
        'train_agent_every': 1,
        'batch_size': 32,
        'buffer_capacity': 1_000_000,
        'seed': 1234,
        'num_cat': 601,
        'cat_min': -300,
        'cat_max': 300,
        'learning_rate': 0.00025,  # TODO this needs to be a schedule
        'num_actors': 1,
        'num_training_steps': 1_000_000,
        'min_buffer_length': 1_000,
        'env_name': 'PongNoFrameskip-v4',
    }

    # env = main.muzero_wrap_atari('BreakoutNoFrameskip-v4')
    # obs = env.reset()
    # hist = environments.history_buffer.create_history(31, 18, (96, 96, 3))
    # hist = environments.history_buffer.reset(hist, obs)
    # agent_obs = (hist.obs, hist.a)
    agent_obs = jnp.array(0)
    muzero_params, muzero_comps = build_muzero(config)
    mcts_testing(agent_obs, muzero_params, muzero_comps, config)