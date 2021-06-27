from typing import List

import gym

import jax.random as jrng
import numpy as np

import common
from actors import parallel_actor
from actors.parallel_actor import ReplayConsumable, Reset
from environments import history_buffer
from networks.jitted_muzero_functions import MuZeroActor
from networks.muzero_def import MuZeroParams


def get_return(
        traj: List[ReplayConsumable]
):
    total_return = 0
    for elem in traj:
        if isinstance(elem, Reset):
            continue
        else:
            total_return += elem.r
    return total_return


def evaluate_agent(
        env: gym.Env,
        actor: MuZeroActor,
        key: jrng.PRNGKey,
        muzero_params: MuZeroParams,
        config: common.Config,
):
    all_returns = []
    hist = history_buffer.create_history(config['num_stack'], config['num_actions'], config['obs_shape'])

    for _ in range(config['num_evals']):
        obs = env.reset()
        hist = history_buffer.reset(hist, obs)
        returns = 0
        for ts in range(config['max_steps']):
            key, actor_key = jrng.split(key)
            a_vec, _, _ = actor(muzero_params, actor_key, (hist.obs[None, :], hist.a[None, :]), 1)
            a = a_vec[0]
            obs, r, done, info = env.step(a)
            hist = history_buffer.step(hist, obs, a, r, done)
            returns += r
            if done:
                break
        all_returns.append(returns)
    return np.mean(all_returns)
