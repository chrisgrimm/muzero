import common
import mcts
import jax, jax.numpy as jnp, jax.random as jrng
import numpy as np
from pprint import pprint

# dummy environment.
# 3 states, tabular representation. fixed dynamics
from networks import actor_network
from muzero import init_muzero

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
    next_s_idx = jax.lax.cond(a == 0,
                        lambda _: jnp.argmax(P_a1[s_idx, :]),
                        lambda _: jnp.argmax(P_a2[s_idx, :]),
                        operand=None)
    return jax.nn.one_hot(next_s_idx, 3)



dummy_config = {
    'obs_shape': (1,),
    'embedding_size': 3,
    'num_actions': 2,
    'num_simulations': 3,
    'gamma': 0.99,
}

key = jrng.PRNGKey(1234)
key, *keys = jrng.split(key, 4)

muzero = init_muzero(
    key=key,
    embed=dummy_embed,
    reward=dummy_reward,
    value=dummy_value,
    policy=dummy_policy,
    dynamics=dummy_dynamics,
    config=dummy_config,
)

dummy_obs = jnp.array(0)

mcts_params = mcts.init_mcts_params(muzero, keys[0], dummy_obs, dummy_config)
rollout = mcts.rollout_to_leaf(mcts_params, dummy_config)
mcts_params = mcts.expand_leaf(mcts_params, muzero, keys[1], rollout, dummy_config)
mcts_params = mcts.backup(mcts_params, rollout, dummy_config)

rollout2 = mcts.MCTSRollout(
    nodes=jnp.array([0, 1, 1]),
    actions=jnp.array([0, 0, 0]),
    valid=jnp.array([True, True, False])
)
mcts_params = mcts.expand_leaf(mcts_params, muzero, keys[2], rollout2, dummy_config)
mcts_params = mcts.backup(mcts_params, rollout2, dummy_config)