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

import os
from collections import deque
from functools import partial
from pathlib import Path

import absl
import absl.flags
import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from data import Dataset, DM2Gym, RandSampler, RLUPDataset
from diffusion.constants import (
  DATASET,
  DATASET_ABBR_MAP,
  DATASET_MAP,
  ENV,
  ENV_MAP,
  ENVNAME_MAP,
)
from diffusion.diffusion import (
  GaussianDiffusion,
  LossType,
  ModelMeanType,
  ModelVarType,
)
from diffusion.dql import DiffusionQL
from diffusion.hps import hyperparameters
from diffusion.nets import Critic, DiffusionPolicy, GaussianPolicy, Value
from utilities.jax_utils import batch_to_jax, next_rng
from utilities.replay_buffer import get_d4rl_dataset
from utilities.sampler import TrajSampler
from utilities.utils import (
  Timer,
  WandBLogger,
  define_flags_with_default,
  get_user_flags,
  norm_obs,
  prefix_metrics,
  set_random_seed,
)
from viskit.logging import logger, setup_logger

FLAGS_DEF = define_flags_with_default(
  algo="DiffQL",
  # algo="DiffusionQL",
  type="model-free",
  env="walker2d-medium-replay-v2",
  dataset='d4rl',
  rl_unplugged_task_class='control_suite',
  max_traj_length=1000,
  save_model=False,
  seed=42,
  batch_size=256,
  reward_scale=1,
  reward_bias=0,
  clip_action=0.999,
  encoder_arch="64-64",
  policy_arch="256-256-256",
  qf_arch="256-256-256",
  orthogonal_init=False,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  algo_cfg=DiffusionQL.get_default_config(),
  n_epochs=2000,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=10,
  logging=WandBLogger.get_default_config(),
  qf_layer_norm=False,
  policy_layer_norm=False,
  activation="mish",
  obs_norm=False,
  act_method='',
  sample_method='ddpm',
  policy_temp=1.0,
  norm_reward=False,
)


class SamplerPolicy(object):

  def __init__(
    self, policy, qf=None, mean=0, std=1, ensemble=False, act_method='ddpm'
  ):
    self.policy = policy
    self.qf = qf
    self.mean = mean
    self.std = std
    self.num_samples = 50
    self.act_method = act_method

  def update_params(self, params):
    self.params = params
    return self

  @partial(jax.jit, static_argnames=("self", "deterministic"))
  def act(self, params, rng, observations, deterministic):
    return self.policy.apply(
      params["policy"], rng, observations, deterministic, repeat=None
    )

  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def ensemble_act(
    self, params, rng, observations, deterministic, num_samples
  ):
    rng, key = jax.random.split(rng)
    actions = self.policy.apply(
      params["policy"], key, observations, deterministic, repeat=num_samples
    )
    q1 = self.qf.apply(params['qf1'], observations, actions)
    q2 = self.qf.apply(params['qf2'], observations, actions)
    q = jnp.minimum(q1, q2)

    idx = jax.random.categorical(rng, q)
    return jnp.take(actions, idx, axis=-2)

  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def ddpmensemble_act(
    self, params, rng, observations, deterministic, num_samples
  ):
    rng, key = jax.random.split(rng)
    actions = self.policy.apply(
      params["policy"],
      rng,
      observations,
      deterministic,
      repeat=num_samples,
      method=self.policy.ddpm_sample,
    )
    q1 = self.qf.apply(params['qf1'], observations, actions)
    q2 = self.qf.apply(params['qf2'], observations, actions)
    q = jnp.minimum(q1, q2)

    idx = jax.random.categorical(rng, q)
    return jnp.take(actions, idx, axis=-2)

  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def dpmensemble_act(
    self, params, rng, observations, deterministic, num_samples
  ):
    rng, key = jax.random.split(rng)
    actions = self.policy.apply(
      params["policy"],
      rng,
      observations,
      deterministic,
      repeat=num_samples,
      method=self.policy.dpm_sample,
    )
    q1 = self.qf.apply(params['qf1'], observations, actions)
    q2 = self.qf.apply(params['qf2'], observations, actions)
    q = jnp.minimum(q1, q2)

    idx = jax.random.categorical(rng, q)
    return jnp.take(actions, idx, axis=-2)

  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def dpm_act(self, params, rng, observations, deterministic, num_samples):
    return self.policy.apply(
      params["policy"],
      rng,
      observations,
      deterministic,
      method=self.policy.dpm_sample,
    )

  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def ddim_act(self, params, rng, observations, deterministic, num_samples):
    return self.policy.apply(
      params["policy"],
      rng,
      observations,
      deterministic,
      method=self.policy.ddim_sample,
    )

  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def ddpm_act(self, params, rng, observations, deterministic, num_samples):
    return self.policy.apply(
      params["policy"],
      rng,
      observations,
      deterministic,
      method=self.policy.ddpm_sample,
    )

  def __call__(self, observations, deterministic=False):
    observations = (observations - self.mean) / self.std
    actions = getattr(self, f"{self.act_method}_act")(
      self.params, next_rng(), observations, deterministic, self.num_samples
    )
    if isinstance(actions, tuple):
      actions = actions[0]
    assert jnp.all(jnp.isfinite(actions))
    return jax.device_get(actions)


class DiffusionTrainer:

  def __init__(self):
    self._cfgs = absl.flags.FLAGS
    self._algo = DiffusionQL
    self._algo_type = 'DiffusionQL'

    self._cfgs.algo_cfg.max_grad_norm = hyperparameters[self._cfgs.env]['gn']
    self._cfgs.algo_cfg.lr_decay_steps = \
      self._cfgs.n_epochs * self._cfgs.n_train_step_per_epoch

    if self._cfgs.activation == 'mish':
      act_fn = lambda x: x * jnp.tanh(jax.nn.softplus(x))
    else:
      act_fn = getattr(jax.nn, self._cfgs.activation)

    self._act_fn = act_fn

    self._variant = get_user_flags(self._cfgs, FLAGS_DEF)
    for k, v in self._cfgs.algo_cfg.items():
      self._variant[f"algo.{k}"] = v

    # get high level env
    env_name_full = self._cfgs.env
    for scenario_name in ENV_MAP:
      if scenario_name in env_name_full:
        self._env = ENV_MAP[scenario_name]
        break
    else:
      raise NotImplementedError

  def train(self):
    self._setup()

    act_methods = self._cfgs.act_method.split('-')
    viskit_metrics = {}
    recent_returns = {method: deque(maxlen=10) for method in act_methods}
    best_returns = {method: -float('inf') for method in act_methods}
    for epoch in range(self._cfgs.n_epochs):
      metrics = {"epoch": epoch}

      with Timer() as train_timer:
        for _ in tqdm.tqdm(range(self._cfgs.n_train_step_per_epoch)):
          batch = batch_to_jax(self._dataset.sample())
          metrics.update(prefix_metrics(self._agent.train(batch), "agent"))

      with Timer() as eval_timer:
        if epoch == 0 or (epoch + 1) % self._cfgs.eval_period == 0:

          for method in act_methods:
            # TODO: merge these two
            self._sampler_policy.act_method = \
              method or self._cfgs.sample_method + "ensemble"
            if self._cfgs.sample_method == 'ddim':
              self._sampler_policy.act_method = "ensemble"
            trajs = self._eval_sampler.sample(
              self._sampler_policy.update_params(self._agent.train_params),
              self._cfgs.eval_n_trajs,
              deterministic=True,
              obs_statistics=(self._obs_mean, self._obs_std, self._obs_clip),
            )

            post = "" if len(act_methods) == 1 else "_" + method
            metrics["average_return" +
                    post] = np.mean([np.sum(t["rewards"]) for t in trajs])
            metrics["average_traj_length" +
                    post] = np.mean([len(t["rewards"]) for t in trajs])
            metrics["average_normalizd_return" + post] = cur_return = np.mean(
              [
                self._eval_sampler.env.get_normalized_score(
                  np.sum(t["rewards"])
                ) for t in trajs
              ]
            )
            recent_returns[method].append(cur_return)
            metrics["average_10_normalized_return" +
                    post] = np.mean(recent_returns[method])
            metrics["best_normalized_return" +
                    post] = best_returns[method] = max(
                      best_returns[method], cur_return
                    )
            metrics["done" +
                    post] = np.mean([np.sum(t["dones"]) for t in trajs])

          if self._cfgs.save_model:
            save_data = {
              "agent": self._agent,
              "variant": self._variant,
              "epoch": epoch
            }
            self._wandb_logger.save_pickle(save_data, f"model_{epoch}.pkl")

      metrics["train_time"] = train_timer()
      metrics["eval_time"] = eval_timer()
      metrics["epoch_time"] = train_timer() + eval_timer()
      self._wandb_logger.log(metrics)
      viskit_metrics.update(metrics)
      logger.record_dict(viskit_metrics)
      logger.dump_tabular(with_prefix=False, with_timestamp=False)

    # save model
    if self._cfgs.save_model:
      save_data = {
        "agent": self._agent,
        "variant": self._variant,
        "epoch": epoch
      }
      self._wandb_logger.save_pickle(save_data, "model_final.pkl")

  def _setup(self):

    set_random_seed(self._cfgs.seed)
    # setup logger
    self._wandb_logger = self._setup_logger()

    # setup dataset and eval_sample
    self._dataset, self._eval_sampler = self._setup_dataset()

    # setup policy
    self._policy = self._setup_policy()
    self._policy_dist = GaussianPolicy(
      self._action_dim, temperature=self._cfgs.policy_temp
    )

    # setup Q-function
    self._qf = self._setup_qf()
    self._vf = self._setup_vf()

    # setup agent
    self._agent = self._algo(
      self._cfgs.algo_cfg, self._policy, self._qf, self._vf, self._policy_dist
    )

    # setup sampler policy
    self._sampler_policy = SamplerPolicy(self._agent.policy, self._agent.qf)

  def _setup_qf(self):
    qf = Critic(
      self._observation_dim,
      self._action_dim,
      to_arch(self._cfgs.qf_arch),
      use_layer_norm=self._cfgs.qf_layer_norm,
      act=self._act_fn,
      orthogonal_init=self._cfgs.orthogonal_init,
    )
    return qf

  def _setup_vf(self):
    vf = Value(
      self._observation_dim,
      to_arch(self._cfgs.qf_arch),
      use_layer_norm=self._cfgs.qf_layer_norm,
      act=self._act_fn,
      orthogonal_init=self._cfgs.orthogonal_init,
    )
    return vf

  def _setup_policy(self):
    gd = GaussianDiffusion(
      num_timesteps=self._cfgs.algo_cfg.num_timesteps,
      schedule_name=self._cfgs.algo_cfg.schedule_name,
      model_mean_type=ModelMeanType.EPSILON,
      model_var_type=ModelVarType.FIXED_SMALL,
      loss_type=LossType.MSE,
      min_value=-self._max_action,
      max_value=self._max_action,
    )
    policy = DiffusionPolicy(
      diffusion=gd,
      observation_dim=self._observation_dim,
      action_dim=self._action_dim,
      arch=to_arch(self._cfgs.policy_arch),
      time_embed_size=self._cfgs.algo_cfg.time_embed_size,
      use_layer_norm=self._cfgs.policy_layer_norm,
      sample_method=self._cfgs.sample_method,
      dpm_steps=self._cfgs.algo_cfg.dpm_steps,
      dpm_t_end=self._cfgs.algo_cfg.dpm_t_end,
    )

    return policy

  def _setup_logger(self):
    env_name_high = ENVNAME_MAP[self._env]
    env_name_full = self._cfgs.env
    dataset_name_abbr = DATASET_ABBR_MAP[self._cfgs.dataset]

    logging_configs = self._cfgs.logging
    logging_configs["project"] = f"{self._cfgs.algo}-{env_name_high}-" + \
      f"{dataset_name_abbr}-{self._cfgs.algo_cfg.loss_type}"
    wandb_logger = WandBLogger(
      config=logging_configs, variant=self._variant, env_name=env_name_full
    )
    setup_logger(
      variant=self._variant,
      exp_id=wandb_logger.experiment_id,
      seed=self._cfgs.seed,
      base_log_dir=self._cfgs.logging.output_dir,
      include_exp_prefix_sub_dir=False,
    )

    return wandb_logger

  def _setup_d4rl(self):
    eval_sampler = TrajSampler(
      gym.make(self._cfgs.env), self._cfgs.max_traj_length
    )

    norm_reward = self._cfgs.norm_reward
    if 'antmaze' in self._cfgs.env:
      norm_reward = False

    dataset = get_d4rl_dataset(
      eval_sampler.env,
      self._cfgs.algo_cfg.nstep,
      self._cfgs.algo_cfg.discount,
      norm_reward=norm_reward,
    )
    dataset["rewards"] = dataset[
      "rewards"] * self._cfgs.reward_scale + self._cfgs.reward_bias
    dataset["actions"] = np.clip(
      dataset["actions"], -self._cfgs.clip_action, self._cfgs.clip_action
    )

    if self._env == ENV.Kitchen or self._env == ENV.Adroit or self._env == ENV.Antmaze:
      if self._cfgs.obs_norm:
        self._obs_mean = dataset["observations"].mean()
        self._obs_std = dataset["observations"].std()
        self._obs_clip = 10
      norm_obs(dataset, self._obs_mean, self._obs_std, self._obs_clip)

      if self._env == ENV.Antmaze:
        if self._cfgs.algo_cfg.loss_type == 'IQL':
          dataset["rewards"] -= 1
        else:
          dataset["rewards"] = (dataset["rewards"] - 0.5) * 4
      else:
        min_r, max_r = np.min(dataset["rewards"]), np.max(dataset["rewards"])
        dataset["rewards"] = (dataset["rewards"] - min_r) / (max_r - min_r)
        dataset["rewards"] = (dataset["rewards"] - 0.5) * 2

    # set sampler
    dataset = Dataset(dataset)
    sampler = RandSampler(dataset.size(), self._cfgs.batch_size)
    dataset.set_sampler(sampler)

    return dataset, eval_sampler

  def _setup_rlup(self):
    path = Path(__file__).absolute().parent.parent / 'data'
    dataset = RLUPDataset(
      self._cfgs.rl_unplugged_task_class,
      self._cfgs.env,
      str(path),
      batch_size=self._cfgs.batch_size,
      action_clipping=self._cfgs.clip_action,
    )

    env = DM2Gym(dataset.env)
    eval_sampler = TrajSampler(env, max_traj_length=self._cfgs.max_traj_length)

    return dataset, eval_sampler

  def _setup_dataset(self):
    self._obs_mean = 0
    self._obs_std = 1
    self._obs_clip = np.inf

    dataset_type = DATASET_MAP[self._cfgs.dataset]

    if dataset_type == DATASET.D4RL:
      dataset, eval_sampler = self._setup_d4rl()
    elif dataset_type == DATASET.RLUP:
      dataset, eval_sampler = self._setup_rlup()
    else:
      raise NotImplementedError

    self._observation_dim = eval_sampler.env.observation_space.shape[0]
    self._action_dim = eval_sampler.env.action_space.shape[0]
    self._max_action = float(eval_sampler.env.action_space.high[0])

    if self._cfgs.algo_cfg.target_entropy >= 0.0:
      action_space = eval_sampler.env.action_space
      self._cfgs.algo_cfg.target_entropy = -np.prod(action_space.shape).item()

    return dataset, eval_sampler


def to_arch(string):
  return tuple(int(x) for x in string.split('-'))


if __name__ == '__main__':

  def main(argv):
    trainer = DiffusionTrainer()
    trainer.train()
    os._exit(os.EX_OK)

  absl.app.run(main)
