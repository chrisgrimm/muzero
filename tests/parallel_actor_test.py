import time
from abc import ABC

import gym
import ray
from gym.spaces import Box, Discrete
import numpy as np
import jax.random as jrng

from actors import parallel_actor as pa

num_actions = 0


def dummy_make_make_muzero_actor(num_stack):
    def dummy_make_muzero_actor():
        def dummy_actor(params, key, obs, temp):
            img_obs, action_obs = obs
            # make sure that the frame-order of the observations is correct.
            assert np.all(np.max(img_obs[:, :-1], axis=1) <= img_obs[:, -1], axis=0)
            # make sure that the observations have the correct shape.
            assert img_obs.shape[1] == action_obs.shape[1] == num_stack
            return (np.zeros([len(obs[0])], dtype=np.uint8),
                    np.zeros([len(obs[0]), num_actions], dtype=np.float32),
                    np.zeros([len(obs[0])], dtype=np.float32))
        return dummy_actor
    return dummy_make_muzero_actor


class DummyEnv(gym.Env):

    def __init__(self):
        self._count = 0

    def step(self, a):
        self._count += 1
        done = np.random.uniform() < 0.1
        return self._count, self._count, done, {}

    def reset(self):
        self._count = 0
        return self._count

    def observation_space(self):
        return Box(0, np.inf, shape=(), dtype=np.int32)

    def action_space(self):
        return Discrete(n=1)


def test_observation_correctness(num_cycles=100):

    handle = pa.init_runner(
        32,
        lambda: DummyEnv(),
        None,
        dummy_make_make_muzero_actor(config['num_stack']),
        1.0,
        jrng.PRNGKey(1234),
        config)
    cycles = 0
    while cycles < num_cycles:
        trajs, handle = pa.get_if_ready(handle)
        if len(trajs) == 0:
            time.sleep(0.001)
        else:
            print(trajs)
            cycles += 1



if __name__ == '__main__':
    config = {
        'gamma': 0.997,
        'num_stack': 4,
        'obs_shape': (96, 96, 3),
        'embedding_shape': (6, 6, 256),
        'num_actions': 1,  # ?
        'num_simulations': 50,
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
    ray.init(local_mode=True)

    try:
        test_observation_correctness()
    finally:
        ray.shutdown()