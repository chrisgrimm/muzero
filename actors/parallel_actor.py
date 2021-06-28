from typing import NamedTuple, Callable, List, Tuple, Union

import gym
import jax.random as jrng
import numpy as np
import ray

import common
from environments import history_buffer
from networks.jitted_muzero_functions import MuZeroActor
from networks.muzero_functions import MuZeroParams
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
    search_policy: np.ndarray
    search_value: float


ReplayConsumable = Union[Step, Reset]


class ObsInfoEnv(gym.Env):

    def __init__(self, env: gym.Env):
        self._env = env

    def step(self, action):
        obs, r, a, info = self._env.step(action)
        return obs, r, a, {'obs': np.array(obs), **info}

    def reset(self):
        return self._env.reset()

    def observation_space(self):
        return self._env.observation_space()

    def action_space(self):
        return self._env.action_space()


@ray.remote(num_gpus=1)
class ParallelTrajectoryRunner:

    def __init__(
            self,
            num_parallel: int,
            env_fn: Callable[[], gym.Env],
            muzero_params: MuZeroParams,
            make_actor: Callable[[], MuZeroActor],
            temperature: float,
            key: jrng.PRNGKey,
            config: common.Config,
    ):
        # os.environ["MKL_NUM_THREADS"] = "1"
        self._num_parallel = num_parallel
        self._trajs: List[List[ReplayConsumable]] = [[] for _ in range(num_parallel)]

        mod_env_fn = lambda: ObsInfoEnv(env_fn())

        self._env = SubprocVecEnv([mod_env_fn for _ in range(num_parallel)])
        self._histories = [history_buffer.create_history(config['num_stack']-1, config['num_actions'], (96, 96, 3))
                           for _ in range(num_parallel)]
        self._make_actor = make_actor
        self._actor = make_actor()
        self._muzero_params = muzero_params
        self._key = key
        self._temperature = temperature
        self._num_actions = config['num_actions']

        self._obs_vec = self._env.reset()


        for i, obs in enumerate(self._obs_vec):
            self._histories[i] = history_buffer.reset(self._histories[i], obs)

        actor_inp = (np.stack([hist.obs for hist in self._histories]),
                     np.stack([hist.a for hist in self._histories]))
        a_vec, pi_vec, value_vec, _ = self._actor(self._muzero_params, key, actor_inp, self._temperature)
        self._a_vec = a_vec
        for i, (obs, pi, v) in enumerate(zip(self._obs_vec, pi_vec, value_vec)):
            self._trajs[i].append(Reset(obs, pi, v))

    def get_trajs(self) -> List[List[ReplayConsumable]]:
        to_emit = []
        j = 0
        while not to_emit:
            self._key, key = jrng.split(self._key, 2)

            # Step using the cached action and update the histories
            self._obs_vec, r_vec, done_vec, info_vec = self._env.step(self._a_vec)
            # history buffer controls what goes into the agent's observation
            for i, (obs, a, r, done, info) in enumerate(zip(self._obs_vec, self._a_vec, r_vec, done_vec, info_vec)):
                obs = info['obs']
                self._histories[i] = history_buffer.step(self._histories[i], obs, a, r, done)
            # Get the next action from the actor.
            actor_inp = (np.stack([hist.obs for hist in self._histories]),
                         np.stack([hist.a for hist in self._histories]))
            a_vec, pi_vec, value_vec, mcts_params_vec = self._actor(self._muzero_params, key, actor_inp, self._temperature)
            if j % 100 == 0:
                print(mcts_params_vec.Q.shape)
                print('mcts_q', np.mean(mcts_params_vec.Q[:, 0, :], axis=0))
            reset_indices = []
            # Store the pi_vec and value_vec into Step objects and update trajectories
            grouped = enumerate(zip(pi_vec, value_vec, self._a_vec, r_vec, self._obs_vec, done_vec, info_vec))
            for i, (pi, value, a, r, obs, done, info) in grouped:
                step_obs = info['obs']
                exp = Step(step_obs, pi, value, a, r, done)
                self._trajs[i].append(exp)
                if done:
                    to_emit.append(self._trajs[i])
                    reset_indices.append(i)
                    self._histories[i] = history_buffer.reset(self._histories[i], obs)
            self._a_vec = a_vec
            # update appropriate quantities for reset indices.
            if len(reset_indices):
                actor_inp = (np.stack([self._histories[i].obs for i in reset_indices]),
                             np.stack([self._histories[i].a for i in reset_indices]))
                reset_a_vec, reset_pi_vec, reset_value_vec, _ = self._actor(
                    self._muzero_params, key, actor_inp, self._temperature)
                for idx, reset_idx in enumerate(reset_indices):
                    self._trajs[reset_idx] = [Reset(self._obs_vec[reset_idx], reset_pi_vec[idx], reset_value_vec[idx])]
                    self._a_vec[reset_idx] = reset_a_vec[idx]
            j += 1
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
        make_actor: Callable[[], MuZeroActor],
        temperature: float,
        key: jrng.PRNGKey,
        config: common.Config,
) -> RunnerHandle:
    runner = ParallelTrajectoryRunner.remote(
        num_parallel, env_fn, muzero_params, make_actor, temperature, key, config)
    task_id = runner.get_trajs.remote()
    return RunnerHandle(runner=runner, task_id=task_id)


def get_if_ready(
        handle: RunnerHandle
) -> Tuple[List[List[ReplayConsumable]], RunnerHandle]:
    done, waiting = ray.wait([handle.task_id], timeout=0)
    if len(done) == 1:
        consumables = ray.get(done[0])
        new_task_id = handle.runner.get_trajs.remote()
        return consumables, RunnerHandle(runner=handle.runner, task_id=new_task_id)
    else:
        return [], handle


def update_muzero_params(
        handle: RunnerHandle,
        muzero_params: MuZeroParams
) -> RunnerHandle:
    handle.runner.update_agent.remote(muzero_params)
    return handle


def update_temperature(
        handle: RunnerHandle,
        temperature: float
) -> RunnerHandle:
    handle.runner.update_temperature.remote(temperature)
    return handle


def compute_priority(
        traj: List[ReplayConsumable],
        t: int,
        config: common.Config,
) -> float:
    n = config['env_rollout_length']
    g = config['gamma']
    returns = 0
    vt = traj[t].search_value
    max_idx = min(t+n+1, len(traj))
    i, tt = 0, t+1
    for i, tt in enumerate(range(t+1, max_idx)):
        returns += g**i * traj[tt].r
    try:
        # this can only fail if t is the last index.
        returns += g**(i+1) * traj[tt].search_value
    except IndexError:
        # terminal states should have 0 value, so passing here is correct.
        pass

    return np.abs(vt - returns)



def feed_buffer(
        trajectories: List[List[ReplayConsumable]],
        buffer: TrajectoryReplayBuffer,
        config: common.Config,
) -> None:
    for traj in trajectories:
        for t in range(len(traj)):
            elem = traj[t]
            priority = compute_priority(traj, t, config)
            if isinstance(elem, Reset):
                buffer.reset(obs=elem.obs,
                             search_v=elem.search_value,
                             search_pi=elem.search_policy,
                             priority=priority)
            else:
                buffer.step(
                    obs=elem.obs,
                    search_v=elem.search_value,
                    search_pi=elem.search_policy,
                    a=elem.a,
                    r=elem.r,
                    done=elem.done,
                    priority=priority
                )
