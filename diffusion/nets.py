# Copyright 2023 Garena Online Private Limited.
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

"""Networks for diffusion policy."""
from functools import partial
from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from diffusion.diffusion import (
  GaussianDiffusion,
  ModelMeanType,
  _extract_into_tensor,
)
from diffusion.dpm_solver import DPM_Solver, NoiseScheduleVP
from utilities.jax_utils import extend_and_repeat


def multiple_action_q_function(forward):

  def wrapped(self, observations, actions, **kwargs):
    multiple_actions = False
    batch_size = observations.shape[0]
    if actions.ndim == 3 and observations.ndim == 2:
      multiple_actions = True
      observations = extend_and_repeat(observations, 1, actions.shape[1])
      observations = observations.reshape(-1, observations.shape[-1])
      actions = actions.reshape(-1, actions.shape[-1])
    q_values = forward(self, observations, actions, **kwargs)
    if multiple_actions:
      q_values = q_values.reshape(batch_size, -1)
    return q_values

  return wrapped


def mish(x):
  return x * jnp.tanh(nn.softplus(x))


def sinusoidal_embedding(timesteps, dim, max_period=10000):
  """
  Create sinusoidal timestep embeddings.
  :param timesteps: a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
  :param dim: the dimension of the output.
  :param max_period: controls the minimum frequency of the embeddings.
  :return: an [N x dim] Tensor of positional embeddings.
  """

  half = dim // 2
  freqs = jnp.exp(
    -jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
  )
  # args = timesteps[:, None] * freqs[None, :]
  args = jnp.expand_dims(timesteps, axis=-1) * freqs[None, :]
  embd = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
  return embd


class TimeEmbedding(nn.Module):
  embed_size: int
  act: callable = mish

  @nn.compact
  def __call__(self, timesteps):
    x = sinusoidal_embedding(timesteps, self.embed_size)
    x = nn.Dense(self.embed_size * 2)(x)
    x = self.act(x)
    x = nn.Dense(self.embed_size)(x)
    return x


class PolicyNet(nn.Module):
  output_dim: int
  arch: Tuple = (256, 256, 256)
  time_embed_size: int = 16
  act: callable = mish
  use_layer_norm: bool = False

  @nn.compact
  def __call__(self, state, action, t):
    time_embed = TimeEmbedding(self.time_embed_size, self.act)(t)
    x = jnp.concatenate([state, action, time_embed], axis=-1)

    for feat in self.arch:
      x = nn.Dense(feat)(x)
      if self.use_layer_norm:
        x = nn.LayerNorm()(x)
      x = self.act(x)

    x = nn.Dense(self.output_dim)(x)
    return x


class DiffusionPolicy(nn.Module):
  diffusion: GaussianDiffusion
  observation_dim: int
  action_dim: int
  arch: Tuple = (256, 256, 256)
  time_embed_size: int = 16
  act: callable = mish
  use_layer_norm: bool = False
  use_dpm: bool = False
  sample_method: str = "ddpm"
  dpm_steps: int = 15
  dpm_t_end: float = 0.001

  def setup(self):
    self.base_net = PolicyNet(
      output_dim=self.action_dim,
      arch=self.arch,
      time_embed_size=self.time_embed_size,
      act=self.act,
      use_layer_norm=self.use_layer_norm,
    )

  def __call__(self, rng, observations, deterministic=False, repeat=None):
    return getattr(self, f"{self.sample_method}_sample"
                  )(rng, observations, deterministic, repeat)

  def ddpm_sample(self, rng, observations, deterministic=False, repeat=None):
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)

    shape = observations.shape[:-1] + (self.action_dim,)

    return self.diffusion.p_sample_loop(
      rng_key=rng,
      model_forward=partial(self.base_net, observations),
      shape=shape,
      clip_denoised=True,
    )

  def dpm_sample(self, rng, observations, deterministic=False, repeat=None):
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)
    noise_clip = True

    shape = observations.shape[:-1] + (self.action_dim,)

    ns = NoiseScheduleVP(
      schedule='discrete', alphas_cumprod=self.diffusion.alphas_cumprod
    )

    def wrap_model(model_fn):

      def wrapped_model_fn(x, t):
        t = (t - 1. / ns.total_N) * ns.total_N

        out = model_fn(x, t)
        # add noise clipping
        if noise_clip:
          t = t.astype(jnp.int32)
          x_w = _extract_into_tensor(
            self.diffusion.sqrt_recip_alphas_cumprod, t, x.shape
          )
          e_w = _extract_into_tensor(
            self.diffusion.sqrt_recipm1_alphas_cumprod, t, x.shape
          )
          max_value = (self.diffusion.max_value + x_w * x) / e_w
          min_value = (self.diffusion.min_value + x_w * x) / e_w

          out = out.clip(min_value, max_value)
        return out

      return wrapped_model_fn

    dpm_sampler = DPM_Solver(
      model_fn=wrap_model(partial(self.base_net, observations)),
      noise_schedule=ns,
      predict_x0=self.diffusion.model_mean_type is ModelMeanType.START_X,
    )
    x = jax.random.normal(rng, shape)
    out = dpm_sampler.sample(x, steps=self.dpm_steps, t_end=self.dpm_t_end)

    return out

  def ddim_sample(self, rng, observations, deterministic=False, repeat=None):
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)

    shape = observations.shape[:-1] + (self.action_dim,)

    return self.diffusion.ddim_sample_loop(
      rng_key=rng,
      model_forward=partial(self.base_net, observations),
      shape=shape,
      clip_denoised=True,
    )

  def forward(self, observations, actions, t):
    return self.base_net(observations, actions, t)

  def loss(self, rng_key, observations, actions, ts):
    terms = self.diffusion.training_losses(
      rng_key,
      model_forward=partial(self.base_net, observations),
      x_start=actions,
      t=ts
    )
    return terms
    noise = jax.random.normal(rng_key, actions.shape, dtype=actions.dtype)
    out = self.base_net(observations, noise, ts * 0)
    return {'loss': jnp.square(out - actions)}

  @property
  def max_action(self):
    return self.diffusion.max_value


class Critic(nn.Module):
  observation_dim: int
  action_dim: int
  arch: Tuple = (256, 256, 256)
  act: callable = mish
  use_layer_norm: bool = False
  orthogonal_init: bool = False

  @nn.compact
  @multiple_action_q_function
  def __call__(self, observations, actions):
    x = jnp.concatenate([observations, actions], axis=-1)

    for feat in self.arch:
      if self.orthogonal_init:
        x = nn.Dense(
          feat,
          kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
          bias_init=jax.nn.initializers.zeros,
        )(
          x
        )
      else:
        x = nn.Dense(feat)(x)
      if self.use_layer_norm:
        x = nn.LayerNorm()(x)
      x = self.act(x)

    if self.orthogonal_init:
      x = nn.Dense(
        1,
        kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
        bias_init=jax.nn.initializers.zeros,
      )(
        x
      )
    else:
      x = nn.Dense(1)(x)
    return jnp.squeeze(x, -1)

  @property
  def input_size(self):
    return self.observation_dim + self.action_dim


class GaussianPolicy(nn.Module):
  action_dim: int
  log_std_min: float = -5.0
  log_std_max: float = 2.0
  temperature: float = 1.0

  @nn.compact
  def __call__(self, mean):
    log_stds = self.param(
      'log_stds', nn.initializers.zeros, (self.action_dim,)
    )
    log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
    return distrax.MultivariateNormalDiag(
      mean, jnp.exp(log_stds * self.temperature)
    )


class Value(nn.Module):
  observation_dim: int
  arch: Tuple = (256, 256, 256)
  act: callable = mish
  use_layer_norm: bool = False
  orthogonal_init: bool = False

  @nn.compact
  def __call__(self, observations):
    x = observations

    for feat in self.arch:
      if self.orthogonal_init:
        x = nn.Dense(
          feat,
          kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
          bias_init=jax.nn.initializers.zeros,
        )(
          x
        )
      else:
        x = nn.Dense(feat)(x)
      if self.use_layer_norm:
        x = nn.LayerNorm()(x)
      x = self.act(x)

    if self.orthogonal_init:
      x = nn.Dense(
        1,
        kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
        bias_init=jax.nn.initializers.zeros,
      )(
        x
      )
    else:
      x = nn.Dense(1)(x)
    return jnp.squeeze(x, -1)

  @property
  def input_size(self):
    return self.observation_dim
