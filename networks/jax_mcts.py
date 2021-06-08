import jax
import jax.ops
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
import common

from networks import actor_network
from typing import NamedTuple

from networks.actor_network import MuZeroParams, MuZeroAgent


class MCTSParams(NamedTuple):
    node_num: int
    transitions: jnp.ndarray
    embeddings: jnp.ndarray
    N: jnp.ndarray
    P: jnp.ndarray
    Q: jnp.ndarray
    R: jnp.ndarray
    V: jnp.ndarray


def init_mcts_params(
        config: common.Config
) -> MCTSParams:
    c = config
    make_store = lambda size, val, t: val * jnp.ones([c['num_simulations'], size], dtype=t)
    return MCTSParams(
        node_num=0,
        transitions=make_store(c['num_actions'], -1, jnp.int32),
        embeddings=make_store(c['embedding_size'], 0, jnp.float32),
        N=make_store(c['num_actions'], 0, jnp.float32),
        P=make_store(c['num_actions'], 0, jnp.float32),
        Q=make_store(c['num_actions'], 0, jnp.float32),
        R=make_store(c['num_actions'], 0, jnp.float32),
        V=make_store(1, 0, jnp.float32)[:, 0]
    )


def next_state(
        mcts_params: MCTSParams,
        node_idx: int,
        action: int
):
    return mcts_params.transitions[node_idx, action]


def is_leaf(
        mcts_params: MCTSParams,
        node_idx: int,
        action: int
):
    return (mcts_params.transitions[node_idx, action] == -1).astype(jnp.float32)


def pick_action(
        mcts_params: MCTSParams,
        node_idx: int,
        c1: float = 1.25,
        c2: float = 19_652
):
    s = node_idx
    Q, P, N = mcts_params.Q, mcts_params.P, mcts_params.N
    term1 = jnp.sqrt(jnp.sum(N[s, :], axis=0, keepdims=True)) / (1 + N[s, :])
    term2 = (jnp.sum(N[s, :], axis=0) + c2 + 1) / c2
    term2 = c1 + jnp.log(term2)
    max_term = Q[node_idx, :] + P[node_idx, :] * term1 * term2
    return jnp.argmax(max_term, axis=0)


def do_simulation(
        mcts_params: MCTSParams,
        config: common.Config
):
    # check if state is leaf.

    def f(x, i):
        node_idx, action, not_leaf = x
        not_leaf = (1 - is_leaf(mcts_params, node_idx, action)) * not_leaf

        new_node_idx = next_state(mcts_params, node_idx, action)
        new_action = pick_action(mcts_params, node_idx)

        node_idx = (not_leaf * new_node_idx + (1 - not_leaf) * node_idx).astype(jnp.int32)
        action = (not_leaf * new_action + (1 - not_leaf) * action).astype(jnp.int32)

        return (node_idx, action, not_leaf), (node_idx, action, not_leaf)

    node_idx = 0
    action = pick_action(mcts_params, node_idx)
    (node_idx, action, _), (path_indices, path_actions, not_leaf) = jax.lax.scan(f, (node_idx, action, 1), jnp.arange(config['num_simulations']))
    return node_idx, action, path_indices, path_actions, not_leaf


def expand_node(
        mcts_params: MCTSParams,
        muzero_agent: MuZeroAgent,
        key: jrng.PRNGKey,
        node_idx: int,
        action: int,
        config: common.Config
) -> MCTSParams:
    muzero_params = muzero_agent.params
    # query model to expand node.
    embedding = mcts_params.embeddings[node_idx]
    key, *keys = jrng.split(key, 5)
    next_embedding = muzero_agent.dynamics.apply(muzero_params.dynamics, keys[0], embedding, action, config)
    reward = muzero_agent.reward.apply(muzero_params.reward, keys[1], next_embedding, config)
    value = muzero_agent.value.apply(muzero_params.value, keys[2], next_embedding, config)
    policy = muzero_agent.policy.apply(muzero_params.policy, keys[3], next_embedding, config)

    return mcts_params._replace(
        embeddings=jax.ops.index_update(
            mcts_params.embeddings, jax.ops.index[mcts_params.node_num, :], next_embedding),
        transitions=jax.ops.index_update(
            mcts_params.transitions, jax.ops.index[mcts_params.node_num, action], mcts_params.node_num),
        R=jax.ops.index_update(mcts_params.R, jax.ops.index[mcts_params.node_num, action], reward),
        P=jax.ops.index_update(mcts_params.P, jax.ops.index[mcts_params.node_num, :], policy),
        V=jax.ops.index_update(mcts_params.V, jax.ops.index[mcts_params.node_num], value),
        node_num=mcts_params.node_num+1
    )


def make_gamma_mat(
        gamma: float,
        num_sim: int
) -> np.ndarray:
    row = lambda o: [gamma ** (t - o) if t - o >= 0 else 0. for t in range(num_sim)]
    return np.array([row(o) for o in range(num_sim)])


def backup(
        mcts_params: MCTSParams,
        path_indices: jnp.ndarray,
        path_actions: jnp.ndarray,
        not_leafs: jnp.ndarray,
        leaf_idx: int,
        config: common.Config
) -> MCTSParams:
    num_sim = config['num_simulations']
    gamma = config['gamma']
    traj_rewards = jax.lax.map(lambda sa: mcts_params.R[sa[0], sa[1]], (path_indices, path_actions))
    gamma_mat, count_mat = make_gamma_mat(gamma, num_sim), make_gamma_mat(1, num_sim)
    intermediary = gamma_mat * traj_rewards[None, :] * not_leafs[None, :]
    discounted_returns = jnp.sum(intermediary, axis=1)  # [num_sims]
    leaf_gamma_powers = jnp.sum(count_mat * not_leafs[None, :], axis=1) # [num_sims]
    discounted_values = gamma**leaf_gamma_powers * mcts_params.V[leaf_idx] # [num_sims]
    g = discounted_returns + discounted_values
    g = not_leafs * g + (1 - not_leafs) * mcts_params.Q[path_indices, path_actions]

    sel_n, sel_q = mcts_params.N[path_indices, path_actions], mcts_params.Q[path_indices, path_actions]
    q_update = (sel_n * sel_q + g) / (sel_n + 1)
    return mcts_params._replace(
        N=jax.ops.index_update(mcts_params.N, jax.ops.index[path_indices, path_actions], sel_n + 1),
        Q=jax.ops.index_update(mcts_params.Q, jax.ops.index[path_indices, path_actions], q_update)
    )














