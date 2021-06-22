import os
from typing import NamedTuple, Callable, List, Tuple, Union

import gym
import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
import ray

import common
from environments import history_buffer
from networks import mcts, muzero_functions
from networks.jitted_muzero_functions import MuZeroActor
from networks.muzero_functions import MuZeroParams, MuZeroComponents
from environments.vec_env.subproc_vec_env import SubprocVecEnv
from replay_buffers.trajectory_replay_buffer import TrajectoryReplayBuffer


class Step(NamedTuple):
    obs: np.ndarray
    search_policy: np.ndarray
    search_value: float
    a: int
    r: float
    done: bool


class Reset(NamedTuple):
    obs: np.ndarray


ReplayConsumable = Union[Step, Reset]


@ray.remote
class ParallelTrajectoryRunner:

    def __init__(
            self,
            num_parallel: int,
            env_fn: Callable[[], gym.Env],
            muzero_params: MuZeroParams,
            actor: MuZeroActor,
            temperature: float,
            key: jrng.PRNGKey,
            num_before: int,
            config: common.Config,
    ):
        # os.environ["MKL_NUM_THREADS"] = "1"
        self._num_parallel = num_parallel
        self._trajs: List[List[ReplayConsumable]] = [[] for _ in range(num_parallel)]

        self._env = SubprocVecEnv([env_fn for _ in range(num_parallel)])
        self._histories = [history_buffer.create_history(num_before, config['num_actions'], (96, 96, 3))
                           for _ in range(num_parallel)]
        self._actor = actor
        self._muzero_params = muzero_params
        self._key = key
        self._temperature = temperature

        self._obs_vec = self._env.reset()
        for i, (obs, policy, value) in enumerate(self._obs_vec):
            self._trajs[i].append(Reset(obs))
            self._histories[i] = history_buffer.reset(self._histories[i], obs)

    def get_trajs(self) -> List[List[ReplayConsumable]]:
        to_emit = []
        while not to_emit:
            self._key, key = jrng.split(self._key, 2)
            actor_inp = (np.stack(hist.obs for hist in self._histories),
                         np.stack(hist.a for hist in self._histories))
            a_vec, pi_vec, value_vec = self._actor(self._muzero_params, key, actor_inp, self._temperature)
            self._obs_vec, r_vec, done_vec, _ = self._env.step(a_vec)
            grouped = enumerate(zip(pi_vec, value_vec, a_vec, r_vec, self._obs_vec, done_vec))
            for i, (pi, value, a, r, obs, done) in grouped:
                exp = Step(obs, pi, value, a, r, done)
                self._trajs[i].append(exp)
                self._histories[i] = history_buffer.step(self._histories[i], obs, a, r, done)
                if done:
                    to_emit.append(self._trajs[i])
                    self._trajs[i] = [Reset(obs)]
                    self._histories[i] = history_buffer.reset(self._histories[i], obs)
        return to_emit

    def update_agent(
            self,
            muzero_params: MuZeroParams
    ) -> None:
        self._muzero_params = muzero_params

    def update_temperature(
            self,
            temperature: float,
    ) -> None:
        self._temperature = temperature


class RunnerHandle(NamedTuple):
    runner: ParallelTrajectoryRunner
    task_id: ray.TaskID


def init_runner(
        num_parallel: int,
        env_fn: Callable[[], gym.Env],
        muzero_params: MuZeroParams,
        actor: MuZeroActor,
        temperature: float,
        key: jrng.PRNGKey,
) -> RunnerHandle:
    runner = ParallelTrajectoryRunner(num_parallel, env_fn, muzero_params, actor, temperature, key)
    task_id = runner.remote.get_trajs()
    return RunnerHandle(runner=runner, task_id=task_id)


def get_if_ready(
        handle: RunnerHandle
) -> Tuple[List[List[ReplayConsumable]], RunnerHandle]:
    done, waiting = ray.wait([handle.task_id], timeout=0)
    if len(done) == 1:
        consumables = ray.get(done[0])
        new_task_id = handle.runner.remote.get_trajs()
        return consumables, RunnerHandle(runner=handle.runner, task_id=new_task_id)
    else:
        return [], handle


def update_muzero_params(
        handle: RunnerHandle,
        muzero_params: MuZeroParams
) -> RunnerHandle:
    handle.runner.update_agent(muzero_params)
    return handle


def update_temperature(
        handle: RunnerHandle,
        temperature: float
) -> RunnerHandle:
    handle.runner.update_temperature(temperature)
    return handle


def feed_buffer(
        trajectories: List[List[ReplayConsumable]],
        buffer: TrajectoryReplayBuffer
) -> None:
    for traj in trajectories:
        for elem in traj:
            if isinstance(elem, Reset):
                buffer.reset(obs=elem.obs)
            else:
                buffer.step(
                    obs=elem.obs,
                    search_v=elem.search_value,
                    search_pi=elem.search_policy,
                    a=elem.a,
                    r=elem.r,
                    done=elem.done
                )
