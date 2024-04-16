from typing import Callable, Sequence, Union

import chex
import flax.linen as nn
import flax.struct
import jax.numpy as jnp
import numpy as np

from ..utils import BatchedState
from . import convolution
from .conv_utils import kernel_d4


def _rescale_range(val, val_range):
    if val_range is None:
        return val
    val_min, val_max = val_range
    val = (val - val_min) / (val_max - val_min)
    return val


class KernelModule(nn.Module):
    feature_count: int

    def __call__(self, val) -> chex.Array:
        raise NotImplementedError


class KernelGauss(KernelModule):
    """Smooth interpolation based on Gaussians.

    Given a value x, the output of the kernel function is an array
    roughly like ``[exp(-(x-s1)^2), exp(-(x-s2)^2), ...]``.
    This can be understood as a smooth approximation to linear
    interpolation based on Gaussians located and positions s1, s2, etc.
    These positions are evenly spaced and fixed, here.

    Args:
        feature_count: Number of positions/Gaussians in kernel.
        val_range: Range of input values to rescale to [0, 1].
        width_factor: Initial width factor of the Gaussians.
            The smaller the factor, the wider the Gaussians.
        adaptive_width: Whether to make the width trainable.
        norm: Whether to keep the sum of the kernel values fixed to 1
            for each input value.
        one_width: Whether the widths, if trainable, can be different
            for each kernel position.
        name: Name of module.
    """
    feature_count: int
    # scale values in this range to [0, 1]
    val_range: tuple[chex.Scalar, chex.Scalar] = None
    width_factor: chex.Numeric = np.log(np.exp(1) - 1)
    adaptive_width: bool = True
    norm: bool = True
    one_width: bool = True
    name: str = None

    @nn.compact
    def __call__(self, val):
        width_shape = () if self.one_width else (self.feature_count,)
        if self.adaptive_width:
            factor = self.param(
                'width_factor',
                nn.initializers.constant(self.width_factor),
                width_shape)
        else:
            factor = self.width_factor
        factor = nn.softplus(factor)
        inverse_width = factor * (self.feature_count - 1)
        # could also make this adaptive
        pos = jnp.linspace(0, 1, self.feature_count)
        val = _rescale_range(val, self.val_range)
        val = - (val - pos)**2 * inverse_width
        out = jnp.exp(val)
        return out / jnp.sum(out) if self.norm else out


class KernelLin(KernelModule):
    """Linear interpolation kernel.

    The output of the model is an array like ``[a1, a2, ...]``
    where either one or two neighboring entries are non-zero.
    The position of the non-zero entry is given by the input value
    with linear interpolation (and thus two entries non-zero)
    if the input value falls between two array indices.
    The value the first and last indices correspond to are either
    0 and 1 or given by the ``minmax`` argument.

    Args:
        feature_count: Number of elements in linear interpolation.
        minmax: Range of input values.
        name: Module name.
    """
    feature_count: int
    val_range: tuple[chex.Scalar, chex.Scalar] = None

    @nn.compact
    def __call__(self, val):
        width = 1 / (self.feature_count - 1)
        pos = np.linspace(0, 1, self.feature_count)
        val = _rescale_range(val, self.val_range)
        val = 1 - jnp.abs(val - pos) / width
        return jnp.maximum(val, 0)


class KernelFourier(KernelModule):
    """Truncated fourier expansion on given interval.

    Given an input x to the model, the output is an array like
            ``[1, sin(2 pi x), cos(2 pi x), sin(4 pi x), ...]``
        (except in a different order).

    Args:
        feature_count: The number of Fourier-terms.
            (This is true if the number is odd. Otherwise, in effect,
            the next smallest odd number is used).
        minmax: The range of input values such that the largest input
            is normalized to 1.
    """
    feature_count: int
    val_range: tuple[chex.Scalar, chex.Scalar] = None

    @nn.compact
    def __call__(self, val):
        freq = jnp.arange(1, (self.feature_count - 1) // 2 + 1)
        val = _rescale_range(val, self.val_range)
        sin = jnp.sin(2 * jnp.pi * freq * val)
        cos = jnp.cos(2 * jnp.pi * freq * val)
        return jnp.concatenate((sin, cos, jnp.array([1.])))


# per-site activations adding features ~ "color" channels

class NonlinearFeatures(nn.Module):

    def apply_feature_map(self, inputs):
        raise NotImplementedError

    def out_channel_count(self, in_channel_count):
        raise NotImplementedError

    @nn.compact
    def __call__(self, inputs, local_coupling, flatten_features=True):
        # we can compute the divergence this way, given the local couplings
        # i.e. the W_xx part of the convolutional kernel, the non-linear
        # feature-map is exclusively site-wise.
        orig_channels = inputs.shape[-1]
        inputs, bwd = nn.vjp(
            lambda mdl, lin: mdl.apply_feature_map(lin), self, inputs, vjp_variables=[])

        if flatten_features:
            local_coupling = local_coupling.reshape(
                orig_channels, orig_channels, -1)

        idc = np.arange(local_coupling.shape[1])
        cotangent_reshape = (*inputs.shape[:-2], 1, 1)
        cotangent = jnp.tile(local_coupling[idc, idc], cotangent_reshape)

        _, inputs_grad = bwd(cotangent)
        divergence = jnp.sum(inputs_grad, np.arange(1, inputs_grad.ndim))

        if flatten_features:
            inputs = inputs.reshape(inputs.shape[:-2] + (-1,))
        return inputs, divergence


class FourierFeatures(NonlinearFeatures):
    feature_count: int
    freq_init: Callable = nn.initializers.uniform(5.0)

    def out_channel_count(self, in_channel_count):
        return in_channel_count * self.feature_count

    def apply_feature_map(self, phi_lin):
        freq = self.param(
            'phi_freq', self.freq_init,
            (phi_lin.shape[-1], self.feature_count))

        features = jnp.einsum('...i,ij->...ij', phi_lin, freq)
        features = jnp.sin(features)

        return features


class PolynomialFeatures(NonlinearFeatures):
    powers: Sequence[int] = (1,)

    def out_channel_count(self, in_channel_count):
        return len(self.powers) * in_channel_count

    def apply_feature_map(self, phi_lin):
        features = jnp.stack([phi_lin ** p for p in self.powers], axis=-1)
        return features


class DivFeatures(NonlinearFeatures):
    feature_count: int = 1
    freq_init: Callable = nn.initializers.uniform(5.0)

    def out_channel_count(self, in_channel_count):
        return in_channel_count * self.feature_count

    def apply_feature_map(self, phi_lin):
        freq = self.param(
            'phi_freq', self.freq_init,
            (phi_lin.shape[-1], self.feature_count))
        freq = jnp.abs(freq)

        features = jnp.einsum('...i,ij->...ij', -phi_lin**2, 1/freq)
        features = jnp.einsum('...ij,...i,ij->...ij', jnp.exp(features), phi_lin, freq)

        return features


class Phi4CNF(nn.Module):
    """Continuous normalizing flow with ODE based on feature kernels.

    The ODE is constructed by tensor contraction & convolution of feature
    vectors/matrices generated from the time and field values.

    Args:
        time_kernel_reduced: If not ``None``, contract the time kernel
            features with a dense matrix of shape
            (time_kernel.feature_count, time_kernel_reduced). This may improve
            training dynamics and, if the number of features is reduced,
            may decrease computational cost.
        features_reduced: If not ``None``, contract the site-wise kernel
            features with a dense matrix of shape
            (n_phi_freq, n_phi_freq_bond). This may improve
            training dynamics and, if the number of features is reduced,
            may decrease computational cost.
        phi_freq_init: Initializer used for the field-feature frequencies.
        kernel_shape: Spacial shape of the convolutional kernel. If None,
            the most general case is used where the kernel shape is exactly
            the lattice shape.
        symmetry: Function that generates the number of orbits and
            the orbit index array. If None, no symmetries (besides
            translation) are enforced.
    """
    # convolution
    kernel_shape: tuple[int, ...]
    kernel_init: nn.initializers.Initializer = nn.initializers.constant(0.)
    symmetry: Callable[[tuple[int, ...]], tuple[int, np.ndarray]] = kernel_d4
    conv_args: dict = flax.struct.field(default_factory=lambda: dict(use_bias=False))
    # time kernel
    time_kernel: KernelModule = KernelFourier(feature_count=91)
    time_kernel_reduced: int = 50
    # field-features
    features: Sequence[NonlinearFeatures] = (FourierFeatures(99), PolynomialFeatures([1]))
    features_reduced: int = 50
    #bias
    bias_features: NonlinearFeatures = DivFeatures(10)

    @nn.compact
    def __call__(self, t, state):
        if type(state) is tuple:
            state = BatchedState(*state)
        else:
            state = BatchedState(state,0)
        aux_channel = False
        kernel_dim = len(self.kernel_shape)
        if state.event_dim == kernel_dim:
            # add an auxiliary channel to phis
            aux_channel = True
            phis = jnp.expand_dims(state.event, -1)
            state = BatchedState(phis, state.log_prob)
        elif state.event_dim != kernel_dim + 1:
            raise RuntimeError(
                f'Got field configurations with shape {state.event_shape} '
                f'while the conv kernel has shape {self.kernel_shape}')

        phis = state.flat_event

        # compute time features
        interp_time = self.time_kernel(t)
        if self.time_kernel_reduced is not None:
            time_superpos = self.param(
                'time_superpos',
                nn.initializers.orthogonal(),
                (self.time_kernel_reduced, self.time_kernel.feature_count)
            ) / self.time_kernel.feature_count
            interp_time = time_superpos @ interp_time

        # compute number of phi-features
        phi_channels = phis.shape[-1]
        feature_counts = [fm.out_channel_count(phi_channels)
                          for fm in self.features]
        total_features = sum(feature_counts)

        # set up convolution
        conv = convolution.EquivConvND(
            features=phi_channels * interp_time.size,
            kernel_size=self.kernel_shape,
            orbit_function=self.symmetry,
            kernel_init=self.kernel_init,
            **self.conv_args,
        )
        in_shape = phis.shape[:-1] + (self.features_reduced or total_features,)
        kernel_params, bias, config = conv.setup_conv(in_shape, phis.dtype)
        kernel_params = kernel_params.init(self)

        # contract with time kernels before convolution
        kernel_params = kernel_params.reshape(
            *kernel_params.shape[:-1], phi_channels, interp_time.size)
        # shape = (conv_orbits, in features, out features)
        kernel_params = kernel_params @ interp_time

        # extract the local-coupling weights
        w00 = kernel_params[0]  # shape = (in features, out features)
        if self.features_reduced is not None:
            freq_superpos = self.param(
                'freq_superpos',
                nn.initializers.orthogonal(),
                (self.features_reduced, total_features)
            ) / total_features

            w00 = jnp.einsum('if,io->fo', freq_superpos, w00)

        # compute non-linear features
        inputs = []  # features that are input to convolution
        divergence = jnp.zeros(())
        feature_start = 0
        for fmap, count in zip(self.features, feature_counts):
            feature_end = feature_start + count
            out, div = fmap(phis, w00[feature_start:feature_end])
            inputs.append(out)
            divergence += div
            feature_start = feature_end

        if self.conv_args['use_bias']:
            mag = jnp.mean(phis, tuple(range(1, len(phis.shape)-1)), keepdims=True)
            a = self.param('bias', nn.initializers.normal(),
                           (self.bias_features.out_channel_count(mag.shape[-1]), 1, len(interp_time)))
            a = a @ interp_time
            out, div = self.bias_features(mag, a)
            bias = out @ a
            divergence += div * mag.squeeze()

        inputs = jnp.concatenate(inputs, axis=-1)
        if self.features_reduced is not None:
            inputs = jnp.einsum('fw,...w->...f', freq_superpos, inputs)

        # actually apply convolution to input features to get vector
        grad_phi = conv.apply_conv(inputs, kernel_params, bias, config)

        grad_phi, div = state.restore_shape(grad_phi, divergence)
        if aux_channel:
            # remove auxiliary channel axis again
            grad_phi = jnp.squeeze(grad_phi, -1)
        return grad_phi, -div

