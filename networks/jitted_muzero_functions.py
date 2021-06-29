from typing import Tuple, Callable

import optax
from jax._src.lax.lax import xc

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
        device: xc.Device,
        config: common.Config,
) -> MuZeroActor:
    @jax.jit(device=device)
    def act(
            muzero_params: MuZeroParams,
            key: jrng.PRNGKey,
            obs: Tuple[jnp.ndarray, jnp.ndarray],
            temperature: jnp.ndarray,
    ):
        batch_sample = jax.vmap(mcts.run_and_get_actor_quantities,
                                (None, None, None, 0, None, None), (0, 0, 0))
        return batch_sample(muzero_params, muzero_comps, key, obs, temperature, config)
    act_j = jax.jit(act, device=device)

    def wrapped(
            muzero_params: MuZeroParams,
            key: jrng.PRNGKey,
            obs: Tuple[jnp.ndarray, jnp.ndarray],
            temperature: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        action, policy, value = act_j(muzero_params, key, tree_map(jnp.array, obs), jnp.array(temperature))
        return np.array(action), np.array(policy), np.array(value)

    return wrapped


def make_multi_gpu_actor(
        muzero_comps: MuZeroComponents,
        num_gpus: int,
        config: common.Config,
) -> MuZeroActor:

    def _split(x):
        splits = [x[i::num_gpus][None, ...] for i in range(num_gpus)]
        return jnp.concatenate(splits, axis=0)

    def _join(x):
        return jnp.concatenate([x[i] for i in range(num_gpus)], axis=0) 

    @jax.jit
    def act(
            muzero_params: MuZeroParams,
            key: jrng.PRNGKey,
            obs: Tuple[jnp.ndarray, jnp.ndarray],
            temperature: jnp.ndarray,
    ):
        batch_sample = jax.vmap(lambda muzero_params, key, obs, temp:  mcts.run_and_get_actor_quantities(muzero_params, muzero_comps, key, obs, temperature, config),
                                (None, None, 0, None), (0, 0, 0, 0))
        batch_sample = jax.pmap(batch_sample, in_axes=(None, None, 0, None), out_axes=(0, 0, 0, 0))
        action, policy, value, mcts_params = batch_sample(muzero_params, key, tree_map(_split, obs), temperature)
        return _join(action), _join(policy), _join(value), tree_map(_join, mcts_params)


    def wrapped(
            muzero_params: MuZeroParams,
            key: jrng.PRNGKey,
            obs: Tuple[jnp.ndarray, jnp.ndarray],
            temperature: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        action, policy, value, mcts_params = act(muzero_params, key, tree_map(jnp.array, obs), jnp.array(temperature))
        return np.array(action), np.array(policy), np.array(value), tree_map(np.array, mcts_params)

    return wrapped

def make_train_function(
        muzero_comps: MuZeroComponents,
        optimizer: optax.GradientTransformation,
        device: xc.Device,
        config: common.Config
):
    def train(muzero_params, opt_state, obs_traj, a_traj, r_traj, search_pi_traj, search_v_traj, importance_weights):
        return muzero_functions.train_muzero(muzero_params, muzero_comps, opt_state, optimizer,
                                             obs_traj, a_traj, r_traj, search_pi_traj, search_v_traj, importance_weights,
                                             config)
    train_j = jax.jit(train, device=device)
    def wrapped(
            muzero_params: MuZeroParams,
            opt_state: optax.OptState,
            obs_traj: np.ndarray,
            a_traj: np.ndarray,
            r_traj: np.ndarray,
            search_pi_traj: np.ndarray,
            search_v_traj: np.ndarray,
            importance_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MuZeroParams, optax.OptState]:
        loss, priorities, r_loss, v_loss, pi_loss, targets, opt_state, muzero_params = train_j(
            muzero_params, opt_state,
            jnp.array(obs_traj), jnp.array(a_traj), jnp.array(r_traj),
            jnp.array(search_pi_traj), jnp.array(search_v_traj), jnp.array(importance_weights)
        )
        return (np.array(loss), np.array(priorities), np.array(r_loss), np.array(v_loss), np.array(pi_loss),
                np.array(targets), muzero_params, opt_state)

    return wrapped
