import haiku as hk
import jax


class ResidualBlock(hk.Module):

    def __init__(self, output_channels, kernel_shape, name=None):
        super().__init__(name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape

    def __call__(self, inp):
        x = hk.Sequential([
            hk.Conv2D(self._output_channels, self._kernel_shape, 1, padding='SAME'),
            hk.LayerNorm([0, 1, 2], True, True),
            jax.nn.relu,
            hk.Conv2D(self._output_channels, self._kernel_shape, 1, padding='SAME'),
            hk.LayerNorm([0, 1, 2], True, True),
        ])(inp)
        x = x + inp
        return jax.nn.relu(x)
