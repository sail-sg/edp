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

"""Jax utils for training."""

import jax
import jax.numpy as jnp


class JaxRNG(object):

  def __init__(self, seed):
    self.rng = jax.random.PRNGKey(seed)

  def __call__(self):
    self.rng, next_rng = jax.random.split(self.rng)
    return next_rng


def init_rng(seed):
  global jax_utils_rng
  jax_utils_rng = JaxRNG(seed)


def next_rng():
  global jax_utils_rng
  return jax_utils_rng()


def extend_and_repeat(tensor, axis, repeat):
  return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val, target):
  return jnp.mean(jnp.square(val - target))


def value_and_multi_grad(fun, n_outputs, argnums=0, has_aux=False):

  def select_output(index):

    def wrapped(*args, **kwargs):
      if has_aux:
        x, *aux = fun(*args, **kwargs)
        return (x[index], *aux)
      else:
        x = fun(*args, **kwargs)
        return x[index]

    return wrapped

  grad_fns = tuple(
    jax.value_and_grad(select_output(i), argnums=argnums, has_aux=has_aux)
    for i in range(n_outputs)
  )

  def multi_grad_fn(*args, **kwargs):
    grads = []
    values = []
    for grad_fn in grad_fns:
      (value, *aux), grad = grad_fn(*args, **kwargs)
      values.append(value)
      grads.append(grad)
    return (tuple(values), *aux), tuple(grads)

  return multi_grad_fn


@jax.jit
def batch_to_jax(batch):
  return jax.tree_map(jax.device_put, batch)
