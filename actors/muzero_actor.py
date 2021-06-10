import os
from typing import NamedTuple, Callable, List

import gym
import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
import ray

import common
from networks import mcts
from networks.actor_network import MuZero
from environments.vec_env.subproc_vec_env import SubprocVecEnv


def make_parallel_muzero_actor(
        config: common.Config,
):
    def act(
            muzero: MuZero,
            key: jrng.PRNGKey,
            obs: jnp.ndarray
    ):
        batch_sample = jax.vmap(mcts.sample_action, (None, None, 0, None), 0)
        return batch_sample(muzero, key, obs, config)

    act = jax.jit(act)

    def wrapped(
            muzero: MuZero,
            key: jrng.PRNGKey,
            obs: np.ndarray,
    ):
        return np.array(act(muzero, key, jnp.array(obs)))

    return wrapped


class Experience(NamedTuple):
    obs: np.ndarray
    a: int
    r: float
    next_obs: np.ndarray


@ray.remote
class ParallelTrajectoryRunner:

    def __init__(
            self,
            num_parallel: int,
            env_fn: Callable[[], gym.Env],
            muzero: MuZero,
            key: jrng.PRNGKey,
            config: common.Config,
    ):
        os.environ["MKL_NUM_THREADS"] = "1"
        self._trajectories = [[] for _ in range(num_parallel)]
        self._env = SubprocVecEnv([env_fn for _ in range(num_parallel)])
        self._obs_vec = self._env.reset()
        self._actor = make_parallel_muzero_actor(config)
        self._muzero = muzero
        self._key = key

    def get_traj(self) -> List[List[Experience]]:
        to_emit = []
        while not to_emit:
            self._key, key = jrng.split(self._key, 2)
            a_vec = self._actor(self._muzero, key, self._obs_vec)
            next_obs_vec, reward_vec, done_vec, _ = self._env.step(a_vec)
            grouped = enumerate(zip(self._obs_vec, a_vec, reward_vec, next_obs_vec, done_vec))
            for i, (obs, a, reward, next_obs, done) in grouped:
                exp = Experience(obs, a, reward, next_obs)
                self._trajectories[i].append(exp)
                if done:
                    to_emit.append(self._trajectories[i])
                    self._trajectories[i] = []
            self._obs_vec = next_obs_vec
        return to_emit

    def update_agent(
            self,
            muzero: MuZero
    ) -> None:
        self._muzero = muzero
