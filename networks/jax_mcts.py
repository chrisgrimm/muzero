import jax
import jax.ops
import jax.numpy as jnp
import jax.random as jrng
import numpy as np

from networks import actor_network


def init_mcts_params(config):
    c = config
    make_store = lambda size, val, t: val * jnp.ones([c['num_simulations'], size], dtype=t)
    return {
        'node_num': 0,
        'transitions': make_store(c['num_actions'], -1, jnp.int32),
        'embeddings': make_store(c['embedding_size'], 0, jnp.float32),
        'N': make_store(c['num_actions'], 0, jnp.float32),
        'P': make_store(c['num_actions'], 0, jnp.float32),
        'Q': make_store(c['num_actions'], 0, jnp.float32),
        'R': make_store(c['num_actions'], 0, jnp.float32),
        'V': make_store(1, 0, jnp.float32)[:, 0],
    }


def next_state(mcts_params, node_idx, action):
    return mcts_params['transitions'][node_idx, action]


def is_leaf(mcts_params, node_idx, action):
    return (mcts_params['transitions'][node_idx, action] == -1).astype(jnp.float32)


def pick_action(mcts_params, node_idx, c1=1.25, c2=19_652):
    s = node_idx
    Q, P, N = mcts_params['Q'], mcts_params['P'], mcts_params['N']
    term1 = jnp.sqrt(jnp.sum(N[s, :], axis=0, keepdims=True)) / (1 + N[s, :])
    term2 = (jnp.sum(N[s, :], axis=0) + c2 + 1) / c2
    term2 = c1 + jnp.log(term2)
    max_term = Q[node_idx, :] + P[node_idx, :] * term1 * term2
    return jnp.argmax(max_term, axis=0)


def do_simulation(mcts_params, config):
    # check if state is leaf.

    def f(x, i):
        node_idx, action, not_leaf = x
        not_leaf = (1 - is_leaf(mcts_params, node_idx, action)) * not_leaf

        new_node_idx = next_state(mcts_params, node_idx, action)
        new_action = pick_action(mcts_params, node_idx)

        node_idx = not_leaf * new_node_idx + (1 - not_leaf) * node_idx
        action = not_leaf * new_action + (1 - not_leaf) * action

        return (node_idx, action, not_leaf), (node_idx, action, not_leaf)

    node_idx = 0
    action = pick_action(mcts_params, node_idx)
    _, (path_indices, path_actions, not_leaf) = jax.lax.scan(f, (node_idx, action, 1), jnp.arange(config['num_iterations']))
    return path_indices, path_actions, not_leaf


def expand_node(mcts_params, muzero_params, node_idx, action, config):
    (embed_params, reward_params, value_params,
     policy_params, dynamics_params) = muzero_params

    # query model to expand node.
    embedding = mcts_params['embeddings'][node_idx]
    next_embedding = actor_network.dynamics_t.apply(dynamics_params, embedding, action, config)
    reward = actor_network.reward_t.apply(reward_params, next_embedding, config)
    value = actor_network.value_t.apply(value_params, next_embedding, config)
    policy = actor_network.policy_t.apply(policy_params, next_embedding, config)

    transitions = mcts_params['transitions']
    embeddings = mcts_params['embeddings']
    rewards = mcts_params['R']
    policies = mcts_params['P']
    new_idx = mcts_params['node_num']
    values = mcts_params['V']

    embeddings = jax.ops.index_update(embeddings, jax.ops.index[new_idx, :], next_embedding)
    transitions = jax.ops.index_update(transitions, jax.ops.index[node_idx, action], new_idx)
    rewards = jax.ops.index_update(rewards, jax.ops.index[node_idx, action], reward)
    policies = jax.ops.index_update(policies, jax.ops.index[new_idx, :], policy)
    values = jax.ops.index_update(values, jax.ops.index[new_idx], value)
    new_idx += 1
    return {
        'node_num': new_idx,
        'transitions': transitions,
        'embeddings': embeddings,
        'N': mcts_params['N'],
        'P': policies,
        'Q': mcts_params['Q'],
        'R': rewards,
        'V': values,
    }


def make_gamma_mat(gamma, num_sim):
    row = lambda o: [gamma ** (t - o) if t - o >= 0 else 0. for t in range(num_sim)]
    return np.array([row(o) for o in range(num_sim)])


def backup(mcts_params, path_indices, path_actions, not_leafs, leaf_idx, config):
    num_sim = config['num_simulations']
    gamma = config['gamma']
    rewards, values, q, n = mcts_params['R'], mcts_params['V'], mcts_params['Q'], mcts_params['N']
    traj_rewards = jax.lax.map(lambda sa: rewards[sa[0], sa[1]], (path_indices, path_actions))
    # [r1 ... rL]
    gamma_mat, count_mat = make_gamma_mat(gamma, num_sim), make_gamma_mat(1, num_sim)
    intermediary = gamma_mat * traj_rewards[None, :] * not_leafs[None, :]
    discounted_returns = jnp.sum(intermediary, axis=1)  # [num_sims]
    leaf_gamma_powers = jnp.sum(count_mat * not_leafs[None, :], axis=1) # [num_sims]
    discounted_values = gamma**leaf_gamma_powers * values[leaf_idx] # [num_sims]
    g = discounted_returns + discounted_values
    g = not_leafs * g + (1 - not_leafs) * q[path_indices, path_actions]

    sel_n, sel_q = n[path_indices, path_actions], q[path_indices, path_actions]
    q_update = (sel_n * sel_q + g) / (sel_n + 1)
    q = jax.ops.index_update(q, jax.ops.index[path_indices, path_actions], q_update)
    n = jax.ops.index_update(n, jax.ops.index[path_indices, path_actions], sel_n + 1)
    return {
        'node_num': mcts_params['node_num'],
        'transitions': mcts_params['transitions'],
        'embeddings': mcts_params['embeddings'],
        'N': n,
        'P': mcts_params['P'],
        'Q': q,
        'R': mcts_params['R'],
        'V': mcts_params['V'],
    }














