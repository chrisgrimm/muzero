from collections import deque
from typing import NamedTuple, Tuple

import gym
import numpy as np
from gym.vector.utils import spaces

from environments.vec_env import atari_wrappers


class EnvHistory(NamedTuple):
    num_before: int
    num_actions: int
    obs: np.ndarray
    a: np.ndarray
    r: np.ndarray
    done: np.ndarray


def step(
        hist: EnvHistory,
        obs: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        done: bool
) -> EnvHistory:
    return hist._replace(
        obs=np.concatenate([hist.obs, obs[None, ...]], axis=0)[-(hist.num_before + 1):],
        a=np.concatenate([hist.a, a[None, ...]], axis=0)[-(hist.num_before + 1):],
        r=np.concatenate([hist.r, r[None, ...]], axis=0)[-(hist.num_before + 1):],
        done=np.concatenate([hist.done, done[None, ...]], axis=0)[-(hist.num_before + 1):]
    )


def reset(
        hist: EnvHistory,
        obs: np.ndarray
) -> EnvHistory:
    return hist._replace(
        obs=np.tile(obs[None, ...], [hist.num_before + 1] + ([1] * len(obs.shape))),
        a=np.random.randint(0, hist.num_actions, size=(hist.num_before + 1,), dtype=np.uint8),
        r=np.zeros((hist.num_before + 1,), dtype=np.float32),
        done=np.zeros((hist.num_before + 1,), dtype=np.bool)
    )


def create_history(
        num_before: int,
        num_actions: int,
        obs_shape: Tuple[int, ...],
) -> EnvHistory:
    return EnvHistory(
        num_before=num_before,
        num_actions=num_actions,
        obs=np.zeros((0,) + obs_shape, dtype=np.uint8),
        a=np.zeros((0,), dtype=np.uint8),
        r=np.zeros((0,), dtype=np.float32),
        done=np.zeros((0,), dtype=np.bool)
    )

