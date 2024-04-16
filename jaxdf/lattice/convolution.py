# Adapted from: https://github.com/google/flax/blob/main/flax/linen/linear.py
#
# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: Modified from the original code to support equivariant convolutions.

import chex
import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from . import conv_utils
from jax import lax
from typing import Sequence, Union, Optional, Callable, List, Tuple, NamedTuple
from flax.linen.linear import PaddingLike, PrecisionLike, Array, Dtype, \
    PRNGKey, Shape, ConvGeneralDilatedT, ShapedArray
from functools import partial


class _ConvConfig(NamedTuple):
    orbits: Optional[Array]
    pads: Optional[list[tuple[int, int]]]
    padding_mode: str
    flat_input_shape: Optional[tuple[int, ...]]
    kernel_shape: tuple[int, ...]
    num_batch_dimensions: int
    input_batch_shape: tuple[int, ...]
    conv_map: Callable


class ParamInit(NamedTuple):
    name: str
    initializer: Callable = None
    init_args: Sequence = []

    def init(self, module: nn.Module):
        if self.initializer is None:
            return None

        return module.param(
            self.name, self.initializer, *self.init_args)


class EquivConvND(nn.Module):
    """Equivariant N-dimensional convolution.

    Note: uses different padding convention for circular boundary
        compared to the flax default convolution.

    Attributes:
        features: number of convolution filters.
        kernel_size: shape of the convolutional kernel. For 1D convolution,
          the kernel size can be passed as an integer. For all other cases, it must
          be a sequence of integers.
        strides: an integer or a sequence of `n` integers, representing the
          inter-window strides (default: 1).
        padding: either the string `'SAME'`, the string `'VALID'`, the string
          `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
          high)` integer pairs that give the padding to apply before and after each
          spatial dimension. A single int is interpeted as applying the same padding
          in all dims and passign a single int in a sequence causes the same padding
          to be used on both sides. `'CAUSAL'` padding for a 1D convolution will
          left-pad the convolution axis, resulting in same-sized output.
        input_dilation: an integer or a sequence of `n` integers, giving the
          dilation factor to apply in each spatial dimension of `inputs`
          (default: 1). Convolution with input dilation `d` is equivalent to
          transposed convolution with stride `d`.
        kernel_dilation: an integer or a sequence of `n` integers, giving the
          dilation factor to apply in each spatial dimension of the convolution
          kernel (default: 1). Convolution with kernel dilation
          is also known as 'atrous convolution'.
        feature_group_count: integer, default 1. If specified divides the input
          features into groups.
        use_bias: whether to add a bias to the output (default: True).
        mask: Optional mask for the weights during masked convolution. The mask must
              be the same shape as the convolution weight matrix.
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see `jax.lax.Precision`
          for details.
        kernel_init: initializer for the convolutional kernel.
        bias_init: initializer for the bias.
    """
    features: int
    kernel_size: Sequence[int]
    orbit_function: Callable = conv_utils.kernel_d4
    strides: Union[None, int, Sequence[int]] = 1
    padding: PaddingLike = 'CIRCULAR'
    input_dilation: Union[None, int, Sequence[int]] = 1
    kernel_dilation: Union[None, int, Sequence[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = False
    mask: Optional[Array] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.linear.default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros_init()
    conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated
    shared_weights: bool = True

    def setup_conv(self, input_shape, dtype):
        if isinstance(self.kernel_size, int):
            raise TypeError('Expected Conv kernel_size to be a'
                            ' tuple/list of integers (eg.: [3, 3]) but got'
                            f' {self.kernel_size}.')
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(x: Optional[Union[int, Sequence[int]]]) -> (
                Tuple[int, ...]):
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        # Combine all input batch dimensions into a single leading batch axis.
        num_batch_dimensions = len(input_shape) - (len(kernel_size) + 1)
        input_batch_shape = ()
        flat_input_shape = None
        if num_batch_dimensions != 1:
            input_batch_shape = input_shape[:num_batch_dimensions]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (
                    (total_batch_size,) + input_shape[num_batch_dimensions:])

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        pads = None
        padding_mode = 'constant'
        padding_lax = nn.linear.canonicalize_padding(self.padding, len(kernel_size))
        if padding_lax == 'CIRCULAR':
            kernel_size_dilated = [
                (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
            ]
            zero_pad: List[Tuple[int, int]] = [(0, 0)]
            # NOTE: flax default is (k-1)//2, k//2 here
            pads = (zero_pad + [(k // 2, (k - 1) // 2) for k in kernel_size_dilated] +
                    [(0, 0)])
            padding_mode = 'wrap'
            padding_lax = 'VALID'
        elif padding_lax == 'CAUSAL':
            if len(kernel_size) != 1:
                raise ValueError(
                    'Causal padding is only implemented for 1D convolutions.')
            left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
            pads = [(0, 0), (left_pad, 0), (0, 0)]
            padding_lax = 'VALID'

        dimension_numbers = nn.linear._conv_dimension_numbers(input_shape)
        in_features = input_shape[-1]

        if self.shared_weights:
            # One shared convolutional kernel for all pixels in the output.
            assert in_features % self.feature_group_count == 0
            kernel_shape = kernel_size + (
                in_features // self.feature_group_count, self.features)
        else:
            if self.feature_group_count != 1:
                raise NotImplementedError(
                    f'`lax.conv_general_dilated_local` does not support '
                    f'`feature_group_count != 1`, got `{self.feature_group_count}`.'
                )

            # Need to know the spatial output shape of a standard convolution to
            # create the unshared convolution kernel.
            conv_output_shape = nn.linear.eval_shape(
                lambda lhs, rhs: self.conv_general_dilated(  # pylint: disable=g-long-lambda
                    lhs=lhs,
                    rhs=rhs,
                    window_strides=strides,
                    padding=padding_lax,
                    dimension_numbers=dimension_numbers,
                    lhs_dilation=input_dilation,
                    rhs_dilation=kernel_dilation,
                ),
                ShapedArray(input_shape, dtype),
                ShapedArray(kernel_size + (in_features, self.features), dtype),
            ).shape

            # One (unshared) convolutional kernel per each pixel in the output.
            kernel_shape = conv_output_shape[1:-1] + (np.prod(kernel_size) *
                                                      in_features, self.features)

        if self.mask is not None and self.mask.shape != kernel_shape:
            raise ValueError('Mask needs to have the same shape as weights. '
                             f'Shapes are: {self.mask.shape}, {kernel_shape}')

        if self.orbit_function is not None:
            orbit_count, orbits = self.orbit_function(kernel_size)
            w_shape = (orbit_count, kernel_shape[-2], kernel_shape[-1])
        else:
            orbits = None
            w_shape = (np.prod(kernel_shape[:-2]),
                       kernel_shape[-2], kernel_shape[-1])
        kernel = ParamInit(
            'kernel', self.kernel_init, [w_shape, self.param_dtype])
        if self.use_bias:
            if self.shared_weights:
                # One bias weight per output channel, shared between pixels.
                bias_shape = (self.features,)
            else:
                # One bias weight per output entry, unshared between pixels.
                bias_shape = conv_output_shape[1:]
            bias = ParamInit('bias', self.bias_init, [bias_shape, self.param_dtype])
        else:
            bias = ParamInit('bias')

        if self.shared_weights:
            conv_map = partial(self.conv_general_dilated,
                               feature_group_count=self.feature_group_count)
        else:
            conv_map = partial(lax.conv_general_dilated_local,
                               filter_shape=kernel_size)

        conv_map = partial(
            conv_map,
            window_strides=strides,
            padding=padding_lax,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=dimension_numbers,
            precision=self.precision,
        )

        config = _ConvConfig(
            orbits, pads, padding_mode, flat_input_shape, kernel_shape,
            num_batch_dimensions, input_batch_shape, conv_map)
        return kernel, bias, config

    def apply_conv(self, inputs, kernel, bias, config: _ConvConfig):
        if isinstance(kernel, ParamInit):
            kernel = kernel.init(self)
        if isinstance(bias, ParamInit):
            bias = bias.init(self)

        if config.flat_input_shape is not None:
            inputs = jnp.reshape(inputs, config.flat_input_shape)

        if config.pads is not None:
            inputs = jnp.pad(inputs, config.pads, mode=config.padding_mode)

        if self.orbit_function is not None:
            kernel = kernel[config.orbits]
        else:
            kernel = kernel.reshape(config.kernel_shape)

        if self.mask is not None:
            kernel *= self.mask

        inputs, kernel, bias = nn.linear.promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        y = config.conv_map(inputs, kernel)

        if self.use_bias:
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y += bias

        if config.num_batch_dimensions != 1:
            output_shape = config.input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)

        return y

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies an equivariant convolution to the inputs.

        Args:
          inputs: input data with dimensions (*batch_dims, spatial_dims...,
            features). This is the channels-last convention, i.e. NHWC for a 2d
            convolution and NDHWC for a 3D convolution. Note: this is different from
            the input convention used by `lax.conv_general_dilated`, which puts the
            spatial dimensions last.
            Note: If the input has more than 1 batch dimension, all batch dimensions
            are flattened into a single dimension for the convolution and restored
            before returning.  In some cases directly vmap'ing the layer may yield
            better performance than this default flattening approach.  If the input
            lacks a batch dimension it will be added for the convolution and removed
            n return, an allowance made to enable writing single-example code.

        Returns:
          The convolved data.
        """
        kernel, bias, config = self.setup_conv(inputs.shape, inputs.dtype)
        y = self.apply_conv(inputs, kernel, bias, config)
        return y
