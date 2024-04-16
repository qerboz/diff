from typing import Any, Sequence

import chex
import diffrax
import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np

from .utils import BatchedState
from .samplers import Sampler
from .solvers import Integrator, DiffraxIntegrator, _solver


class Bijection(nn.Module):
    """Base class for bijections."""

    def forward(self, x: chex.Array, log_prob: chex.Array, **kwargs) \
            -> tuple[chex.Array, chex.Array]:
        """Forward map."""
        raise NotImplementedError

    def reverse(self, x: chex.Array, log_prob: chex.Array, **kwargs) \
            -> tuple[chex.Array, chex.Array]:
        """Inverse map."""
        raise NotImplementedError

    def freeze(self, variables: Any = None, **default_kwargs):
        """Convert to sampler with fixed variables."""
        reverse = self.reverse
        forward = self.forward

        class FrozenBijection(Bijection):
            variables: Any

            def forward(self, x, log_prob, **kwargs):
                kwargs = {**default_kwargs, **kwargs}
                return forward(variables, x, log_prob, **kwargs)

            def reverse(self, x, log_prob, **kwargs):
                kwargs = {**default_kwargs, **kwargs}
                return reverse(variables, x, log_prob, **kwargs)

        return variables


class BijectionSampler(Sampler):
    bijection: Bijection
    prior: Sampler

    @property
    def event_shape(self):
        return self.prior.event_shape

    def sample(self, key, batch_shape=(), **kwargs):
        x, log_prob = self.prior.sample(key, batch_shape)
        return self.bijection.forward(x, log_prob, **kwargs)

    def logpdf(self, value, **kwargs):
        delta = np.zeros(self.prior.as_batched_state(value).batch_shape)
        x, delta = self.bijection.reverse(value, delta, **kwargs)
        log_prob = self.prior.logpdf(x) - delta
        return log_prob

    def forward(self, x, log_prob, **kwargs):
        return self.bijection.forward(x, log_prob, **kwargs)

    def reverse(self, x, log_prob, **kwargs):
        return self.bijection.reverse(x, log_prob, **kwargs)

    def init_model(self, key, init_batch_shape=(), **kwargs):
        variables = self.init(key, key, batch_shape=init_batch_shape, **kwargs, method=self.sample)
        return variables, GenerativeModel(self, kwargs)


@flax.struct.dataclass
class GenerativeModel:
    """Generative model from bijection sampler.

    This is a convenience wrapper around BijectionSampler.
    It mainly serves two purpose:
    1. Explicitly expose the different .apply methods, so we don't have to
       write bs.apply(..., method=bs.sample) etc.
    2. Save a particular set of default keyword arguments (which may still
       be overridden in each function call).
    """
    bijection_sampler: BijectionSampler
    default_args: Any = flax.struct.field(default_factory=dict)

    def forward(self, variables, x, log_prob, **kwargs):
        kwargs = {**self.default_args, **kwargs}
        return self.bijection_sampler.apply(
            variables, x, log_prob, **kwargs,
            method=self.bijection_sampler.forward)

    def reverse(self, variables, x, log_prob, **kwargs):
        kwargs = {**self.default_args, **kwargs}
        return self.bijection_sampler.apply(
            variables, x, log_prob, **kwargs,
            method=self.bijection_sampler.reverse)

    def sample(self, variables, key, batch_shape=(), **kwargs):
        kwargs = {**self.default_args, **kwargs}
        return self.bijection_sampler.apply(
            variables, key, batch_shape, **kwargs,
            method=self.bijection_sampler.sample)

    def log_prob(self, variables, value, **kwargs):
        kwargs = {**self.default_args, **kwargs}
        return self.bijection_sampler.apply(
            variables, value, **kwargs,
            method=self.bijection_sampler.logpdf)


class ContinuousFlow(Bijection):
    vector_field: nn.Module
    integrator: Integrator = DiffraxIntegrator(diffrax.Tsit5())
    dt0: float = 1/50


    # whether to vmap integrator if inputs are batches
    vmap_batch: bool = True
    # whether to scan over batch if inputs are batches
    scan_batch: bool = False
    # (batch) vmap axes of integration args
    # must match **kwarg in forward/reverse function signatures
    args_vmap_axes: dict[int] = None

    def setup(self) -> None:
        def vector_field(vf, t, phis, args):
            return vf(t, phis, **args)

        self.solver = nn.transforms.lift_transform(
            _solver, vector_field, self.integrator)
        axes = (None, None, None, None, 0, self.args_vmap_axes)
        self.solver_vect = jax.vmap(self.solver, axes)

    def forward(self, phis, log_prob, t_start=0.0, t_end=0.9999, **kwargs):
        """Flow field configurations ``phis`` forward."""
        state = BatchedState(phis, log_prob)

        solver = self.solver
        if state.batched:
            if self.scan_batch:
                def _body(_, pl):
                    phi, lp = pl
                    return None, self.forward(phi, lp, t_start, t_end, **kwargs)

                return jax.lax.scan(_body, None, (phis, log_prob))[1]

            start = state.flattened
            if self.vmap_batch:
                solver = self.solver_vect
        else:
            start = state.unbatched

        dt = np.abs(self.dt0) * np.sign(t_end - t_start)
        final = solver(
            self.vector_field, t_start, t_end, dt, start, kwargs
        )

        return state.restore_shape(*final)

    def reverse(self, phis, log_prob, t_start=0.0, t_end=1.0, **kwargs):
        """Flow field configurations ``phis`` backward from t_end to t_start."""
        # just reverse times
        return self.forward(phis, log_prob, t_end, t_start, **kwargs)