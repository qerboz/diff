import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np


def ISM_loss(model, params, key, time, sample):
    noised_sample = model.noise_model.sample(key, time, sample)
    var = model.noise_model.noise_scale(time) ** 2
    div = model.apply(params, time, noised_sample, method = model.div_score)
    l2_score = jnp.sum(model.apply(params, time, noised_sample) ** 2, axis = (1, 2, 3))
    return jnp.mean(var * (l2_score + 2 * div))


def DSM_loss(model, params, key, time, sample, n_mc = 10):
    # s = 0
    # for i in range(n_mc):
    #     noise = random.normal(key, sample.shape)
    #     mean_drift = model.noise_model.mean_drift(time)
    #     std = model.noise_model.noise_scale(time)
    #     s += jnp.mean((std * (model.apply(params, time, mean_drift * sample + std * noise)) + noise) ** 2)
    # return s/n_mc
    noise = random.normal(key, sample.shape)
    mean_drift = model.noise_model.mean_drift(time)
    std = model.noise_model.noise_scale(time)
    return jnp.mean((std * (model.apply(params, time, mean_drift * sample + std * noise)) + noise) ** 2)

def ESM_loss(model, params, key, time, sample, energy, n_mc = 10):
    key, subkey = random.split(key)
    noised_sample = model.noise_model.sample(subkey, time, sample)
    mean_drift = model.noise_model.mean_drift(time)
    std = model.noise_model.noise_scale(time)
    venergy = jax.vmap(energy)
    def logp_t(key, noised_sample):
        denoised_samples = (jnp.stack([noised_sample]*n_mc) - std * 
                        random.normal(key,(n_mc,*noised_sample.shape))) / mean_drift
        
        return jax.scipy.special.logsumexp(- venergy(denoised_samples))
    score_estimate = jax.grad(logp_t, 1)(key, noised_sample)
    return jnp.sum((score_estimate - model.apply(params, time, sample)) ** 2)