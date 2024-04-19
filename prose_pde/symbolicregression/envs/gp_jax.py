from scipy.spatial.distance import cdist
import jax
import jax.numpy as jnp
from jax import random
from einshape import jax_einshape as einshape
from functools import partial

# Define the covariance function
def rbf_kernel(x1, x2, sigma, l):
    """
    Radial basis function kernel
    """
    sq_norm = cdist(x1 / l, x2 / l, metric='sqeuclidean')
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

# Define the covariance function
def rbf_kernel_jax(x1, x2, sigma, l):
    """
    Radial basis function kernel, only support 1D x1 and x2
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    sq_norm = (xx1-xx2)**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

# Define the covariance function
def rbf_sin_kernel_jax(x1, x2, sigma, l):
    """
    suppose x1, x2 in [0,1],
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    sq_norm = (jnp.sin(jnp.pi*(xx1-xx2)))**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

def rbf_circle_kernel_jax(x1, x2, sigma, l):
    """
    suppose x1, x2 in [0,1],
    """
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    xx1_1 = jnp.sin(xx1 * 2 * jnp.pi)
    xx1_2 = jnp.cos(xx1 * 2 * jnp.pi)
    xx2_1 = jnp.sin(xx2 * 2 * jnp.pi)
    xx2_2 = jnp.cos(xx2 * 2 * jnp.pi)
    sq_norm = (xx1_1-xx2_1)**2/(l**2) + (xx1_2-xx2_2)**2/(l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

@partial(jax.jit, static_argnames=('num','kernel', 'k_sigma', 'k_l', 'if_norm', 'norm_const'))
def generate_gaussian_process( ts, init_key=2023,num = 4, kernel =rbf_kernel_jax, k_sigma = 1, k_l = 1,if_norm = True,norm_const =1):
  '''
  ts: 1D array (length,)
  out: Gaussian process samples, 2D array (num, length)
  '''
  key = random.PRNGKey(init_key)
  length = len(ts)
  mean = jnp.zeros((num,length))
  # cov = rbf_kernel(ts[:, None], ts[:, None], sigma=k_sigma, l=k_l)
  cov = kernel(ts, ts, sigma=k_sigma, l=k_l)
  cov = einshape('ii->nii', cov, n = num)
  out = jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(num,), method='svd')
  slope = (out[:, -1] - out[:, 0]) / (ts[-1] - ts[0])
  out -= jnp.outer(slope,ts)


  if if_norm:
  # Normalize each sample to be between 0 and 1
      min_val = jnp.min(out, axis=1, keepdims=True)
      max_val = jnp.max(out, axis=1, keepdims=True)
      out = norm_const *  (out - min_val) / (max_val - min_val)
  return out


# import jax.random
# if __name__ == '__main__':
#
#     # Time series or spatial points
#     ts = jnp.linspace(0, 1, 100)
#
#     # GP parameters
#     num_samples = 5
#     sigma = 1.0
#     length_scale = 0.2
#
#     # Random key
#     key = jax.random.PRNGKey(0)
#
#     # Generate GP samples
#     gp_samples = generate_gaussian_process(key, ts, num_samples, rbf_kernel_jax, k_sigma=sigma, k_l=length_scale)
#
#     pass

    # gp_samples will be a (num_samples, len(ts)) array containing GP samples
