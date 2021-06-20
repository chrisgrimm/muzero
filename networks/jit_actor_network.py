from typing import Tuple

import optax

import actor_network
import common
import jax
import jax.random as jrng
import jax.numpy as jnp
import numpy as np
from networks import mcts

from networks.muzero_def import MuZeroComponents, MuZeroParams


def make_actor(
        muzero_comps: MuZeroComponents,
        config: common.Config,
):
    @jax.jit
    def act(
            muzero_params: MuZeroParams,
            key: jrng.PRNGKey,
            obs: jnp.ndarray,
            temperature: jnp.ndarray,
    ):
        batch_sample = jax.vmap(mcts.run_and_get_policy, (None, None, None, 0, None, None), 0)
        return batch_sample(muzero_params, muzero_comps, key, obs, temperature, config)

    def wrapped(
            muzero_params: MuZeroParams,
            key: jrng.PRNGKey,
            obs: np.ndarray,
            temperature: float,
    ) -> np.ndarray:
        return np.array(act(muzero_params, key, jnp.array(obs), jnp.array(temperature)))

    return wrapped


def make_train_function(
        muzero_comps: MuZeroComponents,
        optimizer: optax.GradientTransformation,
        config: common.Config
):
    @jax.jit
    def train(muzero_params, opt_state, key, obs_traj, a_traj, r_traj):
        return actor_network.train_muzero(muzero_params, muzero_comps, opt_state, optimizer, key,
                                          obs_traj, a_traj, r_traj, config)

    def wrapped(
            muzero_params: MuZeroParams,
            opt_state: optax.OptState,
            key: jrng.PRNGKey,
            obs_traj: np.ndarray,
            a_traj: np.ndarray,
            r_traj: np.ndarray
    ) -> Tuple[np.ndarray, MuZeroParams, optax.OptState]:
        loss, opt_state, muzero_params = train(
            muzero_params, opt_state, key, jnp.ndarray(obs_traj), jnp.ndarray(a_traj), jnp.ndarray(r_traj))
        return np.array(loss), muzero_params, opt_state

    return wrapped
