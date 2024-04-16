from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Union

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ..samplers import IndependentUnitNormal
from ..flow import Bijection, BijectionSampler
from ..utils import BatchedState, cyclic_corr


class SpectrumScaling(Bijection):
    """
    Bijection which is a linear scaling in Fourier space.

    Args:
        event_shape (tuple[int, ...]): The shape of the event.
        channel_axis (Optional[int]): The index of the non-space dimension in the event shape. Default is None.
        exact_logp (bool): Whether to ignore a constant volume factor. Default is False.
    """
    event_shape: tuple[int, ...]
    channel_axis: Optional[int] = None
    exact_logp: bool = False

    def spectrum_function(self, ks):
        # should return
        raise NotImplementedError

    def setup(self):
        shape = self.event_shape
        channel_axis = self.channel_axis
        if channel_axis is not None:
            if channel_axis < 0:
                channel_axis = len(shape) + channel_axis
            shape = tuple(s for i, s in enumerate(shape) if i != channel_axis)

        shape_factor = np.reshape(shape, [-1] + [1] * len(shape))
        # Using reality condition, can eliminate about half of components
        shape = list(shape)[:-1] + [np.floor(shape[-1] / 2) + 1]

        # get frequencies divided by shape as large grid
        # ks[i] is k varying along axis i from 0 to L_i
        ks = np.mgrid[tuple(np.s_[:s] for s in shape)] / shape_factor

        self.scale_factor = self.spectrum_function(ks)

        def scale(r, factors, reverse=False):
            shape = r.shape
            # go to Fourier space
            r = jnp.fft.rfftn(r, shape)
            # scale
            if reverse:
                r = r * factors
            else:
                r = r / factors
            # back to real space
            r = jnp.fft.irfftn(r, shape)
            return r

        unscale = partial(scale, reverse=True)
        if channel_axis is not None:
            scale = jax.vmap(scale, (channel_axis, None), channel_axis)
            unscale = jax.vmap(unscale, (channel_axis, None), channel_axis)

        self.scale = jax.jit(jax.vmap(scale, (0, None), 0))
        self.unscale = jax.jit(jax.vmap(unscale, (0, None), 0))

    def forward(self, x, logp):
        state = BatchedState(x, logp, self.event_shape)
        x = self.scale(state.flat_event, self.scale_factor)
        if self.exact_logp:
            logp = logp + jnp.sum(jnp.log(self.scale_factor))
        return state.restore_shape(x), logp

    def reverse(self, x, logp):
        state = BatchedState(x, logp, self.event_shape)
        x = self.unscale(state.flat_event, self.scale_factor)
        if self.exact_logp:
            logp = logp - jnp.sum(jnp.log(self.scale_factor))
        return state.restore_shape(x), logp


class FreeTheoryScaling(SpectrumScaling):
    """
    Scaling bijection which scales normal samples to free theory spectrum.

    Attributes:
        event_shape (tuple[int, ...]): The shape of the event.
        m2 (Union[Callable, chex.Scalar]): The mass squared parameter. Can be a callable or a scalar value.
        finite_size (bool): Whether to consider finite size effects.
        channel_axis (int): The index of the event_shape that is not a space dimension.
        exact_logp (bool): Whether to include a constant volume factor.
    """
    event_shape: tuple[int, ...]
    m2: Union[Callable, chex.Scalar] = nn.initializers.constant(1.)
    finite_size: bool = True
    channel_axis: int = None  # which index of event_shape is not a space dimension
    exact_logp: bool = False  # whether to include a constant volume factor

    def spectrum_function(self, ks):
        m2 = self.m2
        if callable(m2):
            m2 = self.param('mass_squared', m2, ())

        if self.finite_size:
            sk = m2 + 4 * np.sum(np.sin(np.pi * ks) ** 2, axis=0)
        else:
            sk = m2 + np.sum((2 * np.pi * ks) ** 2, axis=0)
        return jnp.sqrt(2 * sk)


def free_theory_sampler(shape, m2, channel_index=-1, finite=True, exact_logp=False):
    scaling = FreeTheoryScaling(shape, m2, finite, channel_index, exact_logp)
    prior = IndependentUnitNormal(shape)
    return BijectionSampler(scaling, prior).freeze()


@partial(jax.jit, static_argnames=('average',))
def two_point(phis: jnp.ndarray, average: bool = True) -> jnp.ndarray:
    """Estimate ``G(x) = <phi(0) phi(x)>``.

    Translational invariance is assumed, so to improve the estimate we compute
        ``mean_y <phi(y) phi(x+y)>``
    using periodic boundary conditions.

    Args:
        phis: Samples of field configurations of shape
            ``(batch size, L_1, ..., L_d)``.
        average: If false, average over samples is not executed.

    Returns:
        Array of shape ``(L_1, ..., L_d)`` if ``average`` is true, otherwise
        of shape ``(batch size, L_1, ..., L_d)``.
    """
    corr = jax.vmap(cyclic_corr)(phis, phis)
    return jnp.mean(corr, axis=0) if average else corr


@jax.jit
def two_point_central(phis: jnp.ndarray) -> jnp.ndarray:
    """Estimate ``G_c(x) = <phi(0) phi(x)> - <phi(0)> <phi(x)>``.

    Translational invariance is assumed, so to improve the estimate we compute
        ``mean_y <phi(y) phi(x+y)> - <phi(x)> mean_y <phi(x+y)>``
    using periodic boundary conditions.

    Args:
        phis: Samples of field configurations of shape
            ``(batch size, L_1, ..., L_d)``.

    Returns:
        Array of shape ``(L_1, ..., L_d)``.
    """
    phis_mean = jnp.mean(phis, axis=0)
    outer = phis_mean * jnp.mean(phis_mean)

    return two_point(phis, True) - outer


@jax.jit
def correlation_length(G):
    """Estimator for the correlation length.

    Args:
        G: Centered two-point function.

    Returns:
        Scalar. Estimate of correlation length.
    """
    Gs = jnp.mean(G, axis=0)
    arg = (jnp.roll(Gs, 1) + jnp.roll(Gs, -1)) / (2 * Gs)
    mp = jnp.arccosh(arg[1:])
    return 1 / jnp.nanmean(mp)


@partial(jax.jit, static_argnames=('half',))
def phi4_action(phi: jnp.ndarray,
                m2: chex.Scalar = 1,
                lam: chex.Scalar = None,
                half: bool = False) -> jnp.ndarray:
    """Compute the Euclidean action for the scalar phi^4 theory.

    The Lagrangian density is kin(phi) + m2 * phi + l * phi^4

    Args:
        phi: Single field configuration of shape L^d.
        m2: Mass squared term (can be negative).
        lam: Coupling constant for phi^4 term.

    Returns:
        Scalar, the action of the field configuration..
    """
    phis2 = phi ** 2

    # mass term
    a = m2 * phis2

    # kinetic term
    a += sum((jnp.roll(phi, 1, d) - phi) ** 2 for d in range(phi.ndim))

    if half:
        a /= 2

    if lam is not None:
            a += lam * phis2 ** 2

    return jnp.sum(a)


@chex.dataclass
class Phi4Theory:
    """Scalar phi^4 theory."""
    shape: tuple[int, ...]
    m2: chex.Scalar
    lam: chex.Scalar = None
    half: bool = False

    def __init__(self, *, shape, m2, lam=None, half=False):
        self.shape = tuple(shape)
        self.m2 = m2
        self.lam = lam
        self.half = half

    @property
    def lattice_size(self):
        return np.prod(self.shape)

    @property
    def dim(self):
        return len(self.shape)

    def action(self, phis: jnp.ndarray, *,
               m2: chex.Scalar = None,
               lam: chex.Scalar = None
        ) -> jnp.ndarray:
        """Compute the phi^4 action.

        Args:
            phis: Either a single field configuration of shape L^d or
                a batch of those field configurations.
            m2: Mass squared (can be negative).
            lam: Coupling constant for phi^4 term.
            half: Whether to include a 1/2 factor in the (Euclidean)
                lagrangian.

        Returns:
            Either a scalar value or a 1d array of actions for the
            field configuration(s).
        """
        lam = self.lam if lam is None else lam
        m2 = self.m2 if m2 is None else m2

        # check whether phis are a batch or a single sample
        if phis.ndim == self.dim:
            chex.assert_shape(phis, self.shape)
            action = phi4_action(phis, m2, lam, self.half)
            return action
        else:
            chex.assert_shape(phis[0], self.shape)
            act = partial(phi4_action, m2=m2, lam=lam, half=self.half)
            action = jax.vmap(act)(phis)
            return action
