import jax
import jax.ops
import jax.numpy as jnp
import jax.random as jrng
from jax.experimental.host_callback import id_print
import numpy as np
import common

from typing import NamedTuple, Tuple

from networks.muzero_def import MuZeroParams, MuZeroComponents


class MCTSParams(NamedTuple):
    node_num: int
    transitions: jnp.ndarray
    embeddings: jnp.ndarray
    N: jnp.ndarray
    P: jnp.ndarray
    Q: jnp.ndarray
    R: jnp.ndarray
    V: jnp.ndarray


class MCTSRollout(NamedTuple):
    nodes: jnp.ndarray
    actions: jnp.ndarray
    valid: jnp.ndarray


def init_mcts_params(
        muzero_params: MuZeroParams,
        muzero_comps: MuZeroComponents,
        key: jrng.PRNGKey,
        obs: Tuple[jnp.ndarray, jnp.ndarray],
        config: common.Config
) -> MCTSParams:
    key, *keys = jrng.split(key, 2)
    c = config
    s = muzero_comps.embed.apply(muzero_params.embed, keys[0], obs, config)
    base_params = MCTSParams(
        node_num=0,
        transitions=(-1 * jnp.ones((c['num_simulations'], c['num_actions']), dtype=jnp.int32)),
        embeddings=jnp.zeros((c['num_simulations'] + 1, *c['embedding_shape']), dtype=jnp.float32),
        N=jnp.zeros((c['num_simulations'], c['num_actions']), dtype=jnp.float32),
        P=jnp.zeros((c['num_simulations'], c['num_actions']), dtype=jnp.float32),
        Q=jnp.zeros((c['num_simulations'], c['num_actions']), dtype=jnp.float32),
        R=jnp.zeros((c['num_simulations'], c['num_actions']), dtype=jnp.float32),
        V=jnp.zeros((c['num_simulations'] + 1,), dtype=jnp.float32),
    )
    return base_params._replace(
        embeddings=jax.ops.index_update(base_params.embeddings, jax.ops.index[0], s)
    )


def next_state(
        mcts_params: MCTSParams,
        node_idx: int,
        action: int
):
    return mcts_params.transitions[node_idx, action]


def is_valid_node(
        mcts_params: MCTSParams,
        node_idx: int,
        action: int
):
    return mcts_params.transitions[node_idx, action] != -1


def pick_action(
        mcts_params: MCTSParams,
        node_idx: int,
        c1: float = 1.25,
        c2: float = 19_652
):
    term1 = jnp.sqrt(jnp.sum(mcts_params.N[node_idx, :], axis=0, keepdims=True)) / (1 + mcts_params.N[node_idx, :])
    term2 = (jnp.sum(mcts_params.N[node_idx, :], axis=0) + c2 + 1) / c2
    term2 = c1 + jnp.log(term2)
    max_term = mcts_params.Q[node_idx, :] + mcts_params.P[node_idx, :] * term1 * term2
    return jnp.argmax(max_term, axis=0)


def rollout_to_leaf(
        mcts_params: MCTSParams,
        config: common.Config
) -> MCTSRollout:
    # check if state is leaf.

    def mcts_rollout(x, i):
        node_idx, action, is_valid = x

        # log the output regardless of validity.
        out = node_idx, action, is_valid
        is_valid = is_valid & is_valid_node(mcts_params, node_idx, action)

        # come up with new state and action.
        next_node_idx = next_state(mcts_params, node_idx, action)
        next_action = pick_action(mcts_params, next_node_idx)

        return (next_node_idx, next_action, is_valid), out

    node_idx, action = 0, pick_action(mcts_params, 0)
    _, (path_indices, path_actions, is_valid) = jax.lax.scan(
        mcts_rollout, (node_idx, action, True), jnp.arange(config['num_simulations']))
    return MCTSRollout(
        nodes=path_indices,
        actions=path_actions,
        valid=is_valid
    )


def expand_leaf(
        mcts_params: MCTSParams,
        muzero_params: MuZeroParams,
        muzero_comps: MuZeroComponents,
        key: jrng.PRNGKey,
        rollout: MCTSRollout,
        config: common.Config
) -> MCTSParams:
    last_index = jnp.argmax(jnp.where(rollout.valid, jnp.arange(config['num_simulations']), 0))
    node_idx, action = rollout.nodes[last_index], rollout.actions[last_index]
    embedding = mcts_params.embeddings[node_idx]

    key, *keys = jrng.split(key, 5)
    next_embedding = muzero_comps.dynamics.apply(muzero_params.dynamics, keys[0], embedding, action, config)
    unprocessed_reward = muzero_comps.reward.apply(muzero_params.reward, keys[1], next_embedding, config)
    reward = muzero_comps.process_reward(unprocessed_reward)
    # reward = muzero_functions.get_scalar(reward_cat, config)
    unprocessed_value = muzero_comps.value.apply(muzero_params.value, keys[2], next_embedding, config)
    value = muzero_comps.process_value(unprocessed_value)
    # scaled_value = muzero_functions.get_scalar(value_cat, config)
    # value = muzero_functions.invert_target_transform(scaled_value)
    policy = muzero_comps.policy.apply(muzero_params.policy, keys[3], next_embedding, config)

    expanded_idx = mcts_params.node_num + 1
    mcts_params = mcts_params._replace(
        embeddings=jax.ops.index_update(
            mcts_params.embeddings, jax.ops.index[expanded_idx], next_embedding),
        transitions=jax.ops.index_update(
            mcts_params.transitions, jax.ops.index[node_idx, action], expanded_idx),
        R=jax.ops.index_update(mcts_params.R, jax.ops.index[node_idx, action], reward),
        P=jax.ops.index_update(mcts_params.P, jax.ops.index[node_idx], policy),
        V=jax.ops.index_update(mcts_params.V, jax.ops.index[expanded_idx], value),
        node_num=mcts_params.node_num + 1
    )
    return mcts_params


def make_gamma_mat(
        gamma: float,
        num_sim: int
) -> np.ndarray:
    row = lambda o: [gamma ** (t - o) if t - o >= 0 else 0. for t in range(num_sim)]
    return np.array([row(o) for o in range(num_sim)])


def backup(
        mcts_params: MCTSParams,
        rollout: MCTSRollout,
        config: common.Config
) -> MCTSParams:
    num_sim = config['num_simulations']
    gamma = config['gamma']

    last_index = jnp.argmax(jnp.where(rollout.valid, jnp.arange(config['num_simulations']), 0))
    valid = rollout.valid
    valid_2d = valid[None, :] & valid[:, None]

    expanded_idx = mcts_params.transitions[rollout.nodes[last_index], rollout.actions[last_index]]
    traj_rewards = jnp.where(valid, mcts_params.R[rollout.nodes, rollout.actions], 0)

    gamma_mat = jnp.array(make_gamma_mat(gamma, num_sim))
    gamma_mat = jnp.where(valid_2d, gamma_mat, 0)

    discounted_returns = jnp.sum(traj_rewards[None, :] * gamma_mat, axis=1)

    ones_mat = jnp.array(make_gamma_mat(1, num_sim))
    ones_mat = jnp.where(valid_2d, ones_mat, 0)
    gamma_powers = jnp.sum(ones_mat, axis=1)
    value_discounts = jnp.where(gamma_powers > 0, gamma ** gamma_powers, 0)

    discounted_values = value_discounts * mcts_params.V[expanded_idx]

    g = discounted_returns + discounted_values

    sel_n, sel_q = mcts_params.N[rollout.nodes, rollout.actions], mcts_params.Q[rollout.nodes, rollout.actions]
    q_update = jnp.where(valid, (sel_n * sel_q + g) / (sel_n + 1), sel_q)
    n_update = jnp.where(valid, sel_n + 1, sel_n)
    return mcts_params._replace(
        N=jax.ops.index_update(mcts_params.N, jax.ops.index[rollout.nodes, rollout.actions], n_update),
        Q=jax.ops.index_update(mcts_params.Q, jax.ops.index[rollout.nodes, rollout.actions], q_update)
    )

# def print_rollout(
#         rollout: MCTSRollout,
#         mcts_params: MCTSParams,
# ):
#     print('---')
#     for node, action, valid in zip(rollout.nodes, rollout.actions, rollout.valid):
#         if not valid:
#             break
#         next_node = mcts_params.transitions[node, action]
#         print('node', node)
#         print('action', action)
#         print('r', mcts_params.R[node, action])
#         print('s\'', next_node)
#         print('v(s\')', mcts_params.V[next_node])
#     print('backup', mcts_params.Q[rollout.nodes[0]])
#     print('---')


def run_mcts(
        obs: Tuple[jnp.ndarray, jnp.ndarray],
        key: jrng.PRNGKey,
        muzero_params: MuZeroParams,
        muzero_comps: MuZeroComponents,
        config: common.Config,
) -> MCTSParams:

    def f(carry, i):
        mcts_params, key = carry
        key, *new_keys = jrng.split(key, 2)
        rollout = rollout_to_leaf(mcts_params, config)
        mcts_params = expand_leaf(mcts_params, muzero_params, muzero_comps, new_keys[0], rollout, config)
        mcts_params = backup(mcts_params, rollout, config)
        # print_rollout(rollout, mcts_params)
        #mcts_params = id_print(mcts_params.Q[0, :], result=mcts_params)
        return (mcts_params, key), None

    init_key, mcts_key = jrng.split(key)
    mcts_params = init_mcts_params(muzero_params, muzero_comps, init_key, obs, config)
    (mcts_params, _), _ = jax.lax.scan(f, (mcts_params, mcts_key), jnp.arange(config['num_simulations']))
    return mcts_params


def get_policy(
        mcts_params: MCTSParams,
        temperature: jnp.ndarray,
) -> jnp.ndarray:
    T = 1. / temperature
    return mcts_params.N[0, :]**T / jnp.sum(mcts_params.N[0, :]**T, axis=0)


def get_value(
        mcts_params: MCTSParams,
) -> jnp.ndarray:
    return jnp.max(mcts_params.Q[0, :], axis=0)


def sample_action(
        muzero_params: MuZeroParams,
        muzero_comps: MuZeroComponents,
        key: jrng.PRNGKey,
        obs: Tuple[jnp.ndarray, jnp.ndarray],
        temperature: np.ndarray,
        config: common.Config
) -> jnp.ndarray:
    mcts_key, sample_key = jrng.split(key, 2)
    mcts_params = run_mcts(obs, mcts_key, muzero_params, muzero_comps, config)
    policy = get_policy(mcts_params, temperature)
    return jrng.choice(sample_key, config['num_actions'], p=policy)


def run_and_get_actor_quantities(
        muzero_params: MuZeroParams,
        muzero_comps: MuZeroComponents,
        key: jrng.PRNGKey,
        obs: Tuple[jnp.ndarray, jnp.ndarray],
        temperature: jnp.ndarray,
        config: common.Config
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mcts_key, sample_key = jrng.split(key, 2)
    mcts_params = run_mcts(obs, mcts_key, muzero_params, muzero_comps, config)
    policy = get_policy(mcts_params, temperature)
    action = jrng.choice(sample_key, config['num_actions'], p=policy)
    return action, policy, get_value(mcts_params), mcts_params
