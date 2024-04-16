from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn

from .samplers import Sampler, IndependentUnitNormal, IndependentScaledNormal
from .utils import BatchedState

class NoiseModel(nn.Module):
    time_min = 0
    time_max = 1
    
    def setup(self):
        self.final_time_sampler = Sampler

    #drift function in the sde f(t)
    def sde_drift(self,time):
        raise NotImplementedError
    
    #diffusion function in the sde g(t)
    def sde_diffusion(self,time):
        raise NotImplementedError
    
    #drift coefficient of the marginal distribution at time t
    def mean_drift(self, time):
        raise NotImplementedError
    
    #noise scale of the marginal distribution at time t
    def noise_scale(self,time):
        raise NotImplementedError
    
    #sampling at time t
    def sample(self, key, time, initial_sample):
        state = BatchedState(initial_sample, time)
        def sampling_dist(key, time, sample):
            noise = self.noise_scale(time) * random.normal(key, sample.shape)
            return self.mean_drift(time) * sample + noise
        if state.batched:
            key = random.split(key, initial_sample.shape[0])
            sampling_dist = jax.vmap(sampling_dist, in_axes = (0, 0, 0))
        return sampling_dist(key, time, initial_sample)
    
    def __call__(self, key, time, initial_sample):
        return self.sample(key, time, initial_sample)


class VarianceExplodingModel(NoiseModel):
    noise_min: float = 0.01
    noise_max: float = 50
    time_max: float = 1

    def setup(self):
        super().setup()
        self.final_time_sampler = partial(IndependentScaledNormal, standard_deviation = self.noise_scale(1.))

    def sde_drift(self, time):
        return 0.

    def sde_diffusion(self, time):
        return self.noise_min * (self.noise_max / self.noise_min) ** (time / self.time_max) \
               * jnp.sqrt((2 / self.time_max) * jnp.log(self.noise_max / self.noise_min))
    
    def mean_drift(self, time):
        return 1.

    def noise_scale(self,time):
        return self.noise_min * jnp.sqrt((self.noise_max / self.noise_min) ** (2 * time / self.time_max) - 1)


class VariancePreservingModel(NoiseModel):
    beta_min: float = 0.1
    beta_max: float = 20

    def setup(self):
        super().setup()
        self.final_time_sampler = IndependentUnitNormal

    def beta(self, time):
        return self.beta_min + (self.beta_max - self.beta_min) * time / self.time_max
    
    def int_beta(self, time):
        return self.beta_min * time + ((self.beta_max - self.beta_min) / (2 * self.time_max)) * (time ** 2)

    def sde_drift(self, time):
        return - (1 / 2) * self.beta(time)
    
    def sde_diffusion(self, time):
        return jnp.sqrt(self.beta(time))
    
    def mean_drift(self, time):
        return jnp.exp(- (1 / 2) * self.int_beta(time))
    
    def noise_scale(self, time):
        return jnp.sqrt(1 - jnp.exp(- self.int_beta(time)))


class SubVariancePreservingModel(VariancePreservingModel):

    def sde_diffusion(self, time):
        return jnp.sqrt(self.beta(time) * (1 - jnp.exp(- 2 * self.int_beta(time))))
    
    def noise_scale(self, time):
        return super().noise_scale(time) ** 2