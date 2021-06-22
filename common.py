from typing import Union, Mapping, Tuple, Any, List

import numpy as np
import jax.numpy as jnp

Primitive = Union[str, int, float, bool, Tuple['Primitive']]
Config = Mapping[str, Primitive]

PyTree = Union[np.ndarray, Tuple['PyTree', ...], Mapping[Any, 'PyTree'], List['PyTree']]
JaxPyTree = Union[jnp.ndarray, Tuple['PyTree', ...], Mapping[Any, 'PyTree'], List['PyTree']]


EPS = 1e-8