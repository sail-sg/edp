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

import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
from acme.jax.utils import prefetch


def traj_fn(traj_length):
  def step_proc_fn(batch):
    obs = tf.concat(
      list(batch[rlds.OBSERVATION].values), axis=-1
    )
    return {
      rlds.OBSERVATION: obs,
      rlds.REWARD: batch[rlds.REWARD],
      rlds.ACTION: batch[rlds.ACTION],
      rlds.IS_FIRST: batch[rlds.IS_FIRST],
      rlds.IS_LAST: batch[rlds.IS_LAST],
    }

  def make_traj_ds(episode):
    step_data = episode[rlds.STEPS]
    start = tf.random.uniform(shape=(), minval=0, maxval=traj_length, dtype=tf.int64)
    step_data = step_data.map(step_proc_fn).skip(start)
    trajectory = step_data.batch(traj_length, drop_remainder=True)
    return trajectory
  
  return make_traj_ds


class OfflineDataset:
  def __init__(self, domain='rlu_control_suite', task='walker_walk', batch_size=256, episode_shuffle_size=10, traj_length=10, shuffle_num_steps=50000, buffer_size=10) -> None:
    self._domain = domain
    self._task = task
    self._obs_keys = []
    if 'control_suite' in self._domain:
      self._obs_keys.extend(
        ['height', 'orientations', 'velocity']
      )
    else:
      raise NotImplementedError

    self._ds_name = f"{domain}/{task}"
    self._bs = batch_size

    _ds = tfds.load(self._ds_name)['train']
    _ds = _ds.shuffle(episode_shuffle_size).interleave(
      traj_fn(traj_length),
      cycle_length=100,
      block_length=1,
      deterministic=False,
      num_parallel_calls=tf.data.AUTOTUNE
    )
    _ds = _ds.shuffle(
      shuffle_num_steps // traj_length,
      reshuffle_each_iteration=True
    )
    _ds = _ds.batch(batch_size)
    self._ds = iter(_ds)
    self._ds = prefetch(self._ds, buffer_size=buffer_size)
  
  def sample(self):
    # data has shape [B, T, H]
    return tfds.as_numpy(next(self._ds))


class TransitionDataset(OfflineDataset):
  def __init__(self, domain='rlu_control_suite', task='walker_walk', batch_size=256, episode_shuffle_size=10, shuffle_num_steps=50000) -> None:
      super().__init__(domain, task, batch_size, episode_shuffle_size, 2, shuffle_num_steps)
  
  def sample(self):
    seq_data = super().sample()
    print(seq_data)


if __name__ == "__main__":
  off_ds = TransitionDataset()
  sampled_data = off_ds.sample()

  import tqdm
  for _ in tqdm.tqdm(range(100)):
    off_ds.sample()
