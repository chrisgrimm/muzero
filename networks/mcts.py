import jax
import jax.ops
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
import common

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


class MCTSRollout(NamedTuple):
    nodes: jnp.ndarray
    actions: jnp.ndarray
    leaf_flags: jnp.ndarray


def init_mcts_params(
        muzero_agent: MuZeroAgent,
        key: jrng.PRNGKey,
        obs: jnp.ndarray,
        config: common.Config
) -> MCTSParams:
    key, *keys = jrng.split(key, 2)
    c = config
    s = muzero_agent.embed.apply(muzero_agent.params.embed, keys[0], obs, config)
    base_params = MCTSParams(
        node_num=0,
        transitions=(-1 * jnp.ones((c['num_simulations'], c['num_actions']), dtype=jnp.int32)),
        embeddings=jnp.zeros((c['num_simulations'] + 1, c['embedding_size']), dtype=jnp.float32),
        N=jnp.zeros((c['num_simulations'], c['num_actions']), dtype=jnp.float32),
        P=jnp.zeros((c['num_simulations'], c['num_actions']), dtype=jnp.float32),
        Q=jnp.zeros((c['num_simulations'], c['num_actions']), dtype=jnp.float32),
        R=jnp.zeros((c['num_simulations'], c['num_actions']), dtype=jnp.float32),
        V=jnp.zeros((c['num_simulations'] + 1,), dtype=jnp.float32),
    )
    return base_params._replace(
        embeddings=jax.ops.index_update(base_params.embeddings, jax.ops.index[0, :], s)
    )


def next_state(
        mcts_params: MCTSParams,
        node_idx: int,
        action: int
):
    return mcts_params.transitions[node_idx, action]


def is_leaf_node(
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


def rollout_to_leaf(
        mcts_params: MCTSParams,
        config: common.Config
) -> MCTSRollout:
    # check if state is leaf.

    def mcts_rollout(x, i):
        node_idx, action, is_not_leaf = x
        # log the output regardless of leaf status.
        out = node_idx, action, is_not_leaf
        is_not_leaf = (1 - is_leaf_node(mcts_params, node_idx, action)) * is_not_leaf

        # come up with new state and action.
        next_node_idx = next_state(mcts_params, node_idx, action)
        next_node_idx = (is_not_leaf * next_node_idx + (1 - is_not_leaf) * node_idx).astype(jnp.int32)
        next_action = (is_not_leaf * pick_action(mcts_params, next_node_idx) + (1 - is_not_leaf)).astype(jnp.int32)
        return (next_node_idx, next_action, is_not_leaf), out

    node_idx, action = 0, pick_action(mcts_params, 0)
    _, (path_indices, path_actions, not_leaf) = jax.lax.scan(
        mcts_rollout, (node_idx, action, 1.0), jnp.arange(config['num_simulations']))
    return MCTSRollout(
        nodes=path_indices,
        actions=path_actions,
        leaf_flags=not_leaf
    )


def expand_leaf(
        mcts_params: MCTSParams,
        muzero: MuZeroAgent,
        key: jrng.PRNGKey,
        rollout: MCTSRollout,
        config: common.Config
) -> MCTSParams:
    last_index = jnp.argmax(jnp.arange(config['num_simulations']) * rollout.leaf_flags)
    node_idx, action = rollout.nodes[last_index], rollout.actions[last_index]
    embedding = mcts_params.embeddings[node_idx]

    key, *keys = jrng.split(key, 5)
    next_embedding = muzero.dynamics.apply(muzero.params.dynamics, keys[0], embedding, action, config)
    reward = muzero.reward.apply(muzero.params.reward, keys[1], next_embedding, config)
    value = muzero.value.apply(muzero.params.value, keys[2], next_embedding, config)
    policy = muzero.policy.apply(muzero.params.policy, keys[3], next_embedding, config)

    expanded_idx = mcts_params.node_num + 1
    mcts_params = mcts_params._replace(
        embeddings=jax.ops.index_update(
            mcts_params.embeddings, jax.ops.index[expanded_idx, :], next_embedding),
        transitions=jax.ops.index_update(
            mcts_params.transitions, jax.ops.index[node_idx, action], expanded_idx),
        R=jax.ops.index_update(mcts_params.R, jax.ops.index[node_idx, action], reward),
        P=jax.ops.index_update(mcts_params.P, jax.ops.index[node_idx, :], policy),
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

    last_index = jnp.argmax(jnp.arange(num_sim) * rollout.leaf_flags)
    not_leaf = rollout.leaf_flags == 1
    not_leaf_2d = not_leaf[None, :] & not_leaf[:, None]

    expanded_idx = mcts_params.transitions[rollout.nodes[last_index], rollout.actions[last_index]]
    traj_rewards = jnp.where(not_leaf, mcts_params.R[rollout.nodes, rollout.actions], 0)

    gamma_mat = jnp.array(make_gamma_mat(gamma, num_sim))
    gamma_mat = jnp.where(not_leaf_2d, gamma_mat, 0)

    discounted_returns = jnp.sum(traj_rewards[None, :] * gamma_mat, axis=1)

    ones_mat = jnp.array(make_gamma_mat(1, num_sim))
    ones_mat = jnp.where(not_leaf_2d, ones_mat, 0)
    gamma_powers = jnp.sum(ones_mat, axis=1)
    value_discounts = jnp.where(gamma_powers > 0, gamma ** gamma_powers, 0)

    discounted_values = value_discounts * mcts_params.V[expanded_idx]

    g = discounted_returns + discounted_values

    sel_n, sel_q = mcts_params.N[rollout.nodes, rollout.actions], mcts_params.Q[rollout.nodes, rollout.actions]
    q_update = jnp.where(not_leaf, (sel_n * sel_q + g) / (sel_n + 1), sel_q)
    n_update = jnp.where(not_leaf, sel_n + 1, sel_n)
    return mcts_params._replace(
        N=jax.ops.index_update(mcts_params.N, jax.ops.index[rollout.nodes, rollout.actions], n_update),
        Q=jax.ops.index_update(mcts_params.Q, jax.ops.index[rollout.nodes, rollout.actions], q_update)
    )


def run_mcts(
        obs: jnp.ndarray,
        key: jrng.PRNGKey,
        muzero: MuZeroAgent,
        config: common.Config,
) -> MCTSParams:

    def f(carry, i):
        mcts_params, key = carry
        key, *new_keys = jrng.split(key, 2)
        rollout = rollout_to_leaf(mcts_params, config)
        mcts_params = expand_leaf(mcts_params, muzero, new_keys[0], rollout, config)
        mcts_params = backup(mcts_params, rollout, config)
        return (mcts_params, key), None

    init_key, mcts_key = jrng.split(key)
    mcts_params = init_mcts_params(muzero, init_key, obs, config)
    (mcts_params, _), _ = jax.lax.scan(f, (mcts_params, mcts_key), jnp.arange(config['num_simulations']))
    return mcts_params


def get_policy(
        mcts_params: MCTSParams,
) -> jnp.ndarray:
    return mcts_params.N[0, :] / jnp.sum(mcts_params.N[0, :], axis=0)


def sample_action(
        muzero: MuZeroAgent,
        key: jrng.PRNGKey,
        obs: jnp.ndarray,
        config: common.Config
) -> jnp.ndarray:
    mcts_key, sample_key = jrng.split(key, 2)
    mcts_params = run_mcts(obs, mcts_key, muzero, config)
    policy = get_policy(mcts_params)
    return jrng.choice(sample_key, config['num_actions'], p=policy)
