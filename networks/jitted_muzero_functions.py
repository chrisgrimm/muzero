from typing import Tuple, Callable

import optax

from networks import muzero_functions
import common
import jax
import jax.random as jrng
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
from networks import mcts

from networks.muzero_def import MuZeroComponents, MuZeroParams


MuZeroActor = Callable[[MuZeroParams, jrng.PRNGKey, Tuple[np.ndarray, np.ndarray], float], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def make_actor(
        muzero_comps: MuZeroComponents,
        config: common.Config,
) -> MuZeroActor:
    @jax.jit
    def act(
            muzero_params: MuZeroParams,
            key: jrng.PRNGKey,
            obs: Tuple[jnp.ndarray, jnp.ndarray],
            temperature: jnp.ndarray,
    ):
        batch_sample = jax.vmap(mcts.run_and_get_actor_quantities,
                                (None, None, None, 0, None, None), (0, 0, 0))
        return batch_sample(muzero_params, muzero_comps, key, obs, temperature, config)

    def wrapped(
            muzero_params: MuZeroParams,
            key: jrng.PRNGKey,
            obs: Tuple[jnp.ndarray, jnp.ndarray],
            temperature: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        action, policy, value = act(muzero_params, key, tree_map(jnp.array, obs), jnp.array(temperature))
        print('act!')
        return np.array(action), np.array(policy), np.array(value)

    return wrapped


def make_train_function(
        muzero_comps: MuZeroComponents,
        optimizer: optax.GradientTransformation,
        config: common.Config
):
    @jax.jit
    def train(muzero_params, opt_state, obs_traj, a_traj, r_traj, search_pi_traj, search_v_traj, importance_weights):
        return muzero_functions.train_muzero(muzero_params, muzero_comps, opt_state, optimizer,
                                             obs_traj, a_traj, r_traj, search_pi_traj, search_v_traj, importance_weights,
                                             config)

    def wrapped(
            muzero_params: MuZeroParams,
            opt_state: optax.OptState,
            obs_traj: np.ndarray,
            a_traj: np.ndarray,
            r_traj: np.ndarray,
            search_pi_traj: np.ndarray,
            search_v_traj: np.ndarray,
            importance_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, MuZeroParams, optax.OptState]:
        loss, priorities, opt_state, muzero_params = train(
            muzero_params, opt_state,
            jnp.array(obs_traj), jnp.array(a_traj), jnp.array(r_traj),
            jnp.array(search_pi_traj), jnp.array(search_v_traj), jnp.array(importance_weights)
        )
        return np.array(loss), np.array(priorities), muzero_params, opt_state

    return wrapped
