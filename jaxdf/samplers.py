from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import chex
import flax.linen as nn
import flax.struct

from .utils import make_safe_shape, BatchedState


class Sampler(nn.Module):
    """
    Base class for samplers.

    Subclasses must override event_shape!

    Attributes:
        event_shape (tuple[int, ...]): The shape of a single event/sample.
    """

    # Override! Either by field(init=True) or with @property
    event_shape: tuple[int, ...] = flax.struct.field(init=False)

    def as_batched_state(self, val, log_prob: Optional[jnp.ndarray] = None):
        # Expose BatchedState for convenience
        return BatchedState(val, log_prob, self.event_shape)

    def sample(self, key: chex.PRNGKey, batch_shape=(), **kwargs) -> tuple[chex.Array, chex.Array]:
        """
        Generates a random sample.

        Args:
            key (chex.PRNGKey): The random key for generating the sample.
            batch_shape (tuple, optional): The shape of the batch. Defaults to ().
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[chex.Array, chex.Array]: The generated sample and its log probability.
        """
        raise NotImplementedError

    def logpdf(self, value: chex.Array, **kwargs) -> chex.Array:
        """
        Computes the log probability of a given sample.

        Args:
            value (chex.Array): The sample for which to compute the log probability.
            **kwargs: Additional keyword arguments.

        Returns:
            chex.Array: The log probability of the given sample value.
        """
        raise NotImplementedError

    def freeze(self, variables: Any = None, **default_kwargs):
        """Convert to sampler with fixed variables."""
        event_shape = self.event_shape
        logpdf = partial(self.apply, method=self.logpdf)
        sample = partial(self.apply, method=self.sample)

        class FrozenSampler(Sampler):
            variables: Any

            @property
            def event_shape(self):
                return event_shape

            def logpdf(self, value, **kwargs):
                kwargs = {**default_kwargs, **kwargs}
                return logpdf(variables, value, **kwargs)

            def sample(self, key, batch_shape=(), **kwargs):
                kwargs = {**default_kwargs, **kwargs}
                return sample(variables, key, batch_shape, **kwargs)

        return FrozenSampler(variables)

    def __call__(self, key, batch_shape=(), **kwargs):
        # Default function, to support nn.Module.init
        return self.sample(key, batch_shape, **kwargs)


class IndependentUnitNormal(Sampler):
    event_shape: tuple[int, ...]

    def sample(self, key, batch_shape=()):
        batch = make_safe_shape(batch_shape)
        x = jax.random.normal(key, batch + self.event_shape)
        logp = self.logpdf(x)
        return x, logp

    def logpdf(self, value: chex.Array) -> chex.Array:
        logp = jax.scipy.stats.norm.logpdf(value)
        logp = jnp.sum(logp, axis=self.as_batched_state(value).event_axes)
        return logp


class IndependentScaledNormal(IndependentUnitNormal):
    event_shape: tuple[int, ...]
    standard_deviation: float

    def sample(self, key, batch_shape=()):
        x, logp = super().sample(key, batch_shape)
        return self.standard_deviation * x, logp

    def logpdf(self, value: chex.Array) -> chex.Array:
        logp = jax.scipy.stats.norm.logpdf(value / self.standard_deviation)
        logp = jnp.sum(logp, axis=self.as_batched_state(value).event_axes)
        return logp


class EmpiricalSampler(Sampler):
    dataset: chex.Array
    event_shape: tuple[int, ...]

    def sample(self, key, batch_shape=()):
        batch = make_safe_shape(batch_shape)
        if len(batch) == 0:
            batch = (1,)
        x = jax.random.choice(key, self.dataset, batch)
        logp = self.logpdf(x)
        if batch[0] == 1:
            return  x.squeeze(axis=0), logp.squeeze(axis=0)
        return x, logp

    def logpdf(self, value: chex.Array) -> chex.Array:
        set_size = self.dataset.shape[0]
        n_batch_axes = jnp.ndim(value)-len(self.event_shape)
        batch_size = make_safe_shape(value.shape[:n_batch_axes])
        logp = - jnp.log(set_size) * jnp.ones(batch_size)
        return jnp.asarray(logp)