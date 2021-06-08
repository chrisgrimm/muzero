from typing import Union, Mapping, Tuple

Primitive = Union[str, int, float, bool, Tuple['Primitive']]
Config = Mapping[str, Primitive]
