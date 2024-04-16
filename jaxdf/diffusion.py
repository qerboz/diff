import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from .noise import NoiseModel
from .flow import ContinuousFlow, BijectionSampler


class DiffusionModel(nn.Module):
    noise_model: NoiseModel
    vector_field: nn.Module
    event_shape: tuple

    def setup(self):
        self.sampler = self.noise_model.final_time_sampler(self.event_shape)
        self.flow = BijectionSampler(ContinuousFlow(self.vector_field), self.sampler)

    def __call__(self, t, x):
        return self.score(t, x)
    
    def score(self, time, sample):
        if sample.shape[0] > 6:
            sample = (sample,jnp.zeros(sample.shape[0]))
        else:
            sample = (sample,0)
        vf = - self.vector_field(1-time, sample)[0]
        return (- 2 * (vf - self.noise_model.sde_drift(time) * sample[0])
                / (self.noise_model.sde_diffusion(time) ** 2))
    
    def div_score(self, time, sample):
        if sample.shape[0] > 6:
            div_sample = sample[0].size * jnp.ones((sample.shape[0],))
            event_shape = sample.shape[1:]
            sample = (sample,jnp.zeros(sample.shape[0]))
            batch = True
        else:
            div_sample = sample.size
            event_shape = sample.shape
            sample = (sample,0)
            batch = False
        div_vf = - self.vector_field(1-time, sample)[1]
        return (2 * (div_vf + self.noise_model.sde_drift(time) * div_sample)
                / (self.noise_model.sde_diffusion(time) ** 2))
        # def f(x):
        #     x = x.reshape(event_shape)
        #     return self.score(time, x).flatten()
        # div = lambda x: jnp.trace(jax.jacfwd(f)(x.flatten()))
        # if batch:
        #     div = jax.vmap(div)
        # return div(sample[0])

    def sample(self, key, n = 1, **kwargs):
        return self.flow.sample(key, n, **kwargs)