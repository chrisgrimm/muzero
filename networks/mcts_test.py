import common
import jax_mcts
import jax, jax.numpy as jnp, jax.random as jrng
import numpy as np

# dummy environment.
# 3 states, tabular representation. fixed dynamics
from networks import actor_network

P_a1 = jnp.array([[0, 1, 0],
                 [0, 0, 1],
                 [0, 0, 1]], dtype=np.float32)
P_a2 = jnp.array([[0, 0, 1],
                 [0, 0, 1],
                 [0, 0, 1]], dtype=np.float32)

R = jnp.array([0, 0.5, 1.0])


def dummy_embed(x: jnp.ndarray, config: common.Config):
    return jax.nn.one_hot(x, 3)


def dummy_reward(x: jnp.ndarray, config: common.Config):
    return R[jnp.argmax(x)]


def dummy_value(x: jnp.ndarray, config: common.Config):
    return jnp.array(100.0, dtype=jnp.float32)


def dummy_policy(x: jnp.ndarray, config: common.Config):
    return jnp.array([0.5, 0.5], dtype=jnp.float32)


def dummy_dynamics(x: jnp.ndarray, a: jnp.ndarray, config: common.Config):
    s_idx = jnp.argmax(x)
    return jax.lax.cond(a == 0,
                        lambda _: jnp.argmax(P_a1[s_idx, :]),
                        lambda _: jnp.argmax(P_a2[s_idx, :]),
                        operand=None)

dummy_comp = actor_network.MuZeroComponents(
    embed=dummy_embed,
    reward=dummy_reward,
    value=dummy_value,
    policy=dummy_policy,
    dynamics=dummy_dynamics
)

dummy_config = {
    'obs_shape': (1,),
    'embedding_size': 3,
    'num_actions': 2,
    'num_simulations': 1,
}
key = jrng.PRNGKey(1234)
key, *keys = jrng.split(key, 3)
muzero_agent = actor_network.MuZeroAgent(keys[0], dummy_comp, dummy_config)

mcts_params = jax_mcts.init_mcts_params(dummy_config)

node_idx, action, path_indices, path_actions, not_leaf = jax_mcts.do_simulation(mcts_params, dummy_config)
mcts_params = jax_mcts.expand_node(mcts_params, muzero_agent, keys[1], node_idx, action, dummy_config)

