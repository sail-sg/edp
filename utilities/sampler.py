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

"""Agent trajectory samplers."""

import time

import numpy as np

WIDTH = 250
HEIGHT = 200


class StepSampler(object):

  def __init__(self, env, max_traj_length=1000):
    self.max_traj_length = max_traj_length
    self._env = env
    self._traj_steps = 0
    self._current_observation = self.env.reset()

  def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for _ in range(n_steps):
      self._traj_steps += 1
      observation = self._current_observation
      action = policy(
        observation.reshape(1, -1), deterministic=deterministic
      ).reshape(-1)
      next_observation, reward, done, _ = self.env.step(action)
      observations.append(observation)
      actions.append(action)
      rewards.append(reward)
      dones.append(done)
      next_observations.append(next_observation)

      if replay_buffer is not None:
        replay_buffer.add_sample(
          observation, action, reward, next_observation, done
        )

      self._current_observation = next_observation

      if done or self._traj_steps >= self.max_traj_length:
        self._traj_steps = 0
        self._current_observation = self.env.reset()

    return dict(
      observations=np.array(observations, dtype=np.float32),
      actions=np.array(actions, dtype=np.float32),
      rewards=np.array(rewards, dtype=np.float32),
      next_observations=np.array(next_observations, dtype=np.float32),
      dones=np.array(dones, dtype=np.float32),
    )

  @property
  def env(self):
    return self._env


class TrajSampler(object):

  def __init__(self, env, max_traj_length=1000, render=False):
    self.max_traj_length = max_traj_length
    self._env = env
    self._render = render

  def norm_obs(self, obs, obs_statistics):
    obs_mean, obs_std, obs_clip = obs_statistics
    return np.clip((obs - obs_mean) / (obs_std + 1e-6), -obs_clip, obs_clip)

  def sample(
    self,
    policy,
    n_trajs,
    deterministic=False,
    replay_buffer=None,
    obs_statistics=(0, 1, np.inf),
    env_render_fn="render",
  ):
    trajs = []
    for _ in range(n_trajs):
      observations = []
      actions = []
      rewards = []
      next_observations = []
      dones = []

      observation = self.env.reset()
      observation = self.norm_obs(observation, obs_statistics)

      done = False
      while not done:
        action = policy(
          observation.reshape(1, -1), deterministic=deterministic
        ).reshape(-1)
        next_observation, reward, done, _ = self.env.step(action)
        if self._render:
          getattr(self.env, env_render_fn)()
          time.sleep(0.01)

        next_observation = self.norm_obs(next_observation, obs_statistics)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        next_observations.append(next_observation)

        if replay_buffer is not None:
          replay_buffer.add_sample(
            observation, action, reward, next_observation, done
          )

        observation = next_observation

      trajs.append(
        dict(
          observations=np.array(observations, dtype=np.float32),
          actions=np.array(actions, dtype=np.float32),
          rewards=np.array(rewards, dtype=np.float32),
          next_observations=np.array(next_observations, dtype=np.float32),
          dones=np.array(dones, dtype=np.float32),
        )
      )

    return trajs

  @property
  def env(self):
    return self._env
