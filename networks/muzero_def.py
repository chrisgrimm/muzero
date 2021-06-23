from typing import NamedTuple, Callable, Tuple
import jax.numpy as jnp
import jax.random as jrng
import haiku as hk

import common


class MuZeroParams(NamedTuple):
    embed: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    policy: jnp.ndarray
    dynamics: jnp.ndarray


class MuZeroComponents(NamedTuple):
    embed: hk.Transformed
    reward: hk.Transformed
    value: hk.Transformed
    policy: hk.Transformed
    dynamics: hk.Transformed


def init_muzero(
        key: jrng.PRNGKey,
        embed: Callable[[Tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray],
        reward: Callable[[jnp.ndarray], jnp.ndarray],
        value: Callable[[jnp.ndarray], jnp.ndarray],
        policy: Callable[[jnp.ndarray], jnp.ndarray],
        dynamics: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        config: common.Config,
) -> Tuple[MuZeroParams, MuZeroComponents]:

    key, *new_keys = jrng.split(key, 6)

    dummy_obs = (jnp.zeros((config['num_stack'],) + config['obs_shape'], dtype=jnp.float32),
                 jnp.zeros((config['num_stack'],), dtype=jnp.uint8))
    dummy_action = jnp.array(0, dtype=jnp.int32)

    embed_t = hk.transform(embed)
    embed_params = embed_t.init(new_keys[0], dummy_obs, config)
    dummy_state = embed_t.apply(embed_params, None, dummy_obs, config)

    comps = MuZeroComponents(
        embed=embed_t,
        reward=hk.transform(reward),
        value=hk.transform(value),
        policy=hk.transform(policy),
        dynamics=hk.transform(dynamics)
    )

    params = MuZeroParams(
        embed=embed_params,
        reward=comps.reward.init(new_keys[1], dummy_state, config),
        value=comps.value.init(new_keys[2], dummy_state, config),
        policy=comps.policy.init(new_keys[3], dummy_state, config),
        dynamics=comps.dynamics.init(new_keys[4], dummy_state, dummy_action, config)
    )

    return params, comps
