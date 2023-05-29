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

from copy import deepcopy
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from core.core_api import Algo
from diffusion.diffusion import GaussianDiffusion
from utilities.jax_utils import (
  extend_and_repeat,
  mse_loss,
  next_rng,
  value_and_multi_grad,
)


def update_target_network(main_params, target_params, tau):
  return jax.tree_multimap(
    lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
  )


class DiffusionQL(Algo):

  @staticmethod
  def get_default_config(updates=None):
    cfg = ConfigDict()
    cfg.nstep = 1
    cfg.discount = 0.99
    cfg.tau = 0.005
    cfg.policy_tgt_freq = 5
    cfg.num_timesteps = 100
    cfg.schedule_name = 'linear'
    cfg.time_embed_size = 16
    cfg.alpha = 2.  # NOTE 0.25 in diffusion rl but 2.5 in td3
    cfg.use_pred_astart = True
    cfg.max_q_backup = False
    cfg.max_q_backup_topk = 1
    cfg.max_q_backup_samples = 10

    # learning related
    cfg.lr = 3e-4
    cfg.diff_coef = 1.0
    cfg.guide_coef = 1.0
    cfg.lr_decay = False
    cfg.lr_decay_steps = 1000000
    cfg.max_grad_norm = 0.
    cfg.weight_decay = 0.

    cfg.loss_type = 'TD3'
    cfg.sample_logp = False

    cfg.adv_norm = False
    # CRR-related hps
    cfg.sample_actions = 10
    cfg.crr_ratio_upper_bound = 20
    cfg.crr_beta = 1.0
    cfg.crr_weight_mode = 'mle'
    cfg.fixed_std = True
    cfg.crr_multi_sample_mse = False
    cfg.crr_avg_fn = 'mean'
    cfg.crr_fn = 'exp'

    # IQL-related hps
    cfg.expectile = 0.7
    cfg.awr_temperature = 3.0

    # for dpm-solver
    cfg.dpm_steps = 15
    cfg.dpm_t_end = 0.001

    # useless
    cfg.target_entropy = -1
    if updates is not None:
      cfg.update(ConfigDict(updates).copy_and_resolve_references())
    return cfg

  def __init__(self, cfg, policy, qf, vf, policy_dist):
    self.config = self.get_default_config(cfg)
    self.policy = policy
    self.qf = qf
    self.vf = vf
    self.policy_dist = policy_dist
    self.observation_dim = policy.observation_dim
    self.action_dim = policy.action_dim
    self.max_action = policy.max_action
    self.diffusion: GaussianDiffusion = self.policy.diffusion

    self._total_steps = 0
    self._train_states = {}

    policy_params = self.policy.init(
      next_rng(),
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
    )

    def get_lr(lr_decay=False):
      if lr_decay is True:
        return optax.cosine_decay_schedule(
          self.config.lr, decay_steps=self.config.lr_decay_steps
        )
      else:
        return self.config.lr

    def get_optimizer(lr_decay=False, weight_decay=cfg.weight_decay):
      if self.config.max_grad_norm > 0:
        opt = optax.chain(
          optax.clip_by_global_norm(self.config.max_grad_norm),
          optax.adamw(get_lr(lr_decay), weight_decay=weight_decay),
        )
      else:
        opt = optax.adamw(get_lr(), weight_decay=weight_decay)

      return opt

    self._train_states['policy'] = TrainState.create(
      params=policy_params,
      tx=get_optimizer(self.config.lr_decay, weight_decay=0.0),
      apply_fn=None
    )

    policy_dist_params = self.policy_dist.init(
      next_rng(), jnp.zeros((10, self.action_dim))
    )
    self._train_states['policy_dist'] = TrainState.create(
      params=policy_dist_params, tx=get_optimizer(), apply_fn=None
    )

    qf1_params = self.qf.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )
    qf2_params = self.qf.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )

    vf_params = self.vf.init(next_rng(), jnp.zeros((10, self.observation_dim)))

    self._train_states['qf1'] = TrainState.create(
      params=qf1_params, tx=get_optimizer(), apply_fn=None
    )
    self._train_states['qf2'] = TrainState.create(
      params=qf2_params, tx=get_optimizer(), apply_fn=None
    )
    self._train_states['vf'] = TrainState.create(
      params=vf_params,
      tx=get_optimizer(),
      apply_fn=None,
    )
    self._tgt_params = deepcopy(
      {
        'policy': policy_params,
        'qf1': qf1_params,
        'qf2': qf2_params,
        'vf': vf_params,
      }
    )
    model_keys = ['policy', 'qf1', 'qf2', 'vf', 'policy_dist']

    self._model_keys = tuple(model_keys)

  def get_value_loss(self, batch):

    def value_loss_fn(params, tgt_params, rng):
      observations = batch['observations']
      actions = batch['actions']
      rewards = batch['rewards']
      next_observations = batch['next_observations']
      dones = batch['dones']

      # Compute the target Q values (without gradient)
      if self.config.max_q_backup:
        samples = self.config.max_q_backup_samples
        next_action = self.policy.apply(
          tgt_params['policy'], rng, next_observations, repeat=samples
        )
        next_action = jnp.clip(next_action, -self.max_action, self.max_action)
        next_obs_repeat = jnp.repeat(
          jnp.expand_dims(next_observations, axis=1), samples, axis=1
        )
        tgt_q1 = self.qf.apply(tgt_params['qf1'], next_obs_repeat, next_action)
        tgt_q2 = self.qf.apply(tgt_params['qf2'], next_obs_repeat, next_action)

        tk = self.config.max_q_backup_topk
        if tk == 1:
          tgt_q = jnp.minimum(tgt_q1.max(axis=-1), tgt_q2.max(axis=-1))
        else:
          batch_idx = jax.vmap(lambda x, i: x[i], 0)
          tgt_q1_max = batch_idx(tgt_q1, jnp.argsort(tgt_q1, axis=-1)[:, -tk])
          tgt_q2_max = batch_idx(tgt_q2, jnp.argsort(tgt_q2, axis=-1)[:, -tk])
          tgt_q = jnp.minimum(tgt_q1_max, tgt_q2_max)
      else:
        next_action = self.policy.apply(
          tgt_params['policy'], rng, next_observations
        )
        tgt_q1 = self.qf.apply(
          tgt_params['qf1'], next_observations, next_action
        )
        tgt_q2 = self.qf.apply(
          tgt_params['qf2'], next_observations, next_action
        )
        tgt_q = jnp.minimum(tgt_q1, tgt_q2)
      tgt_q = rewards + (1 - dones) * self.config.discount * tgt_q
      tgt_q = jax.lax.stop_gradient(tgt_q)

      # Compute the current Q estimates
      cur_q1 = self.qf.apply(params['qf1'], observations, actions)
      cur_q2 = self.qf.apply(params['qf2'], observations, actions)

      # qf loss
      qf1_loss = mse_loss(cur_q1, tgt_q)
      qf2_loss = mse_loss(cur_q2, tgt_q)

      qf_loss = qf1_loss + qf2_loss
      return (qf1_loss, qf2_loss), locals()

    return value_loss_fn

  def get_diff_terms(self, params, observations, actions, dones, rng):
    rng, split_rng = jax.random.split(rng)
    ts = jax.random.randint(
      split_rng, dones.shape, minval=0, maxval=self.diffusion.num_timesteps
    )
    rng, split_rng = jax.random.split(rng)
    terms = self.policy.apply(
      params["policy"],
      split_rng,
      observations,
      actions,
      ts,
      method=self.policy.loss,
    )
    if self.config.use_pred_astart:
      pred_astart = self.diffusion.p_mean_variance(
        terms["model_output"], terms["x_t"], ts
      )["pred_xstart"]
    else:
      rng, split_rng = jax.random.split(rng)
      pred_astart = self.policy.apply(
        params['policy'], split_rng, observations
      )
    terms["pred_astart"] = pred_astart

    action_dist = self.policy_dist.apply(params['policy_dist'], pred_astart)
    sample = pred_astart
    if self.config.sample_logp:
      rng, split_rng = jax.random.split(rng)
      sample = action_dist.sample(seed=split_rng)
    if self.config.fixed_std:
      action_dist = distrax.MultivariateNormalDiag(
        pred_astart, jnp.ones_like(pred_astart)
      )
    log_prob = action_dist.log_prob(sample)
    terms['sample'] = sample
    terms['action_dist'] = action_dist
    terms['log_p'] = log_prob

    return terms, ts, log_prob

  def get_diff_loss(self, batch):

    def diff_loss(params, rng):
      observations = batch['observations']
      actions = batch['actions']
      dones = batch['dones']

      terms, ts, _ = self.get_diff_terms(
        params, observations, actions, dones, rng
      )
      diff_loss = terms["loss"].mean()
      pred_astart = terms["pred_astart"]

      return diff_loss, terms, ts, pred_astart

    return diff_loss

  @partial(jax.jit, static_argnames=('self', 'policy_tgt_update'))
  def _train_step(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    if self.config.loss_type not in ['TD3', 'CRR', 'IQL']:
      raise NotImplementedError

    return getattr(self, f"_train_step_{self.config.loss_type.lower()}"
                  )(train_states, tgt_params, rng, batch, policy_tgt_update)

  def _train_step_td3(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    value_loss_fn = self.get_value_loss(batch)
    diff_loss_fn = self.get_diff_loss(batch)

    def policy_loss_fn(params, tgt_params, rng):
      observations = batch['observations']

      rng, split_rng = jax.random.split(rng)
      diff_loss, _, _, pred_astart = diff_loss_fn(params, split_rng)

      # Calculate guide loss
      def fn(key):
        q = self.qf.apply(params[key], observations, pred_astart)
        lmbda = self.config.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        policy_loss = -lmbda * q.mean()
        return lmbda, policy_loss

      lmbda, guide_loss = jax.lax.cond(
        jax.random.uniform(rng) > 0.5, partial(fn, 'qf1'), partial(fn, 'qf2')
      )

      policy_loss = diff_loss + self.config.guide_coef * guide_loss
      return (policy_loss,), locals()

    # Calculat q losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_qf), grads_qf = value_and_multi_grad(
      value_loss_fn, 2, has_aux=True
    )(params, tgt_params, rng)

    # Calculat policy losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_policy), grads_policy = value_and_multi_grad(
      policy_loss_fn, 1, has_aux=True
    )(params, tgt_params, rng)

    # Update qf train states
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=grads_qf[0]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=grads_qf[1]['qf2']
    )

    # Update policy train states
    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=grads_policy[0]['policy']
    )

    # Update target parameters
    if policy_tgt_update:
      tgt_params['policy'] = update_target_network(
        train_states['policy'].params, tgt_params['policy'], self.config.tau
      )
    tgt_params['qf1'] = update_target_network(
      train_states['qf1'].params, tgt_params['qf1'], self.config.tau
    )
    tgt_params['qf2'] = update_target_network(
      train_states['qf2'].params, tgt_params['qf2'], self.config.tau
    )

    metrics = dict(
      qf_loss=aux_qf['qf_loss'],
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      cur_q1=aux_qf['cur_q1'].mean(),
      cur_q2=aux_qf['cur_q2'].mean(),
      tgt_q1=aux_qf['tgt_q1'].mean(),
      tgt_q2=aux_qf['tgt_q2'].mean(),
      tgt_q=aux_qf['tgt_q'].mean(),
      policy_loss=aux_policy['policy_loss'],
      guide_loss=aux_policy['guide_loss'],
      diff_loss=aux_policy['diff_loss'],
      lmbda=aux_policy['lmbda'].mean(),
      qf1_grad_norm=optax.global_norm(grads_qf[0]['qf1']),
      qf2_grad_norm=optax.global_norm(grads_qf[1]['qf2']),
      policy_grad_norm=optax.global_norm(grads_policy[0]['policy']),
      qf1_weight_norm=optax.global_norm(train_states['qf1'].params),
      qf2_weight_norm=optax.global_norm(train_states['qf2'].params),
      policy_weight_norm=optax.global_norm(train_states['policy'].params),
    )

    return train_states, tgt_params, metrics

  def _train_step_crr(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    value_loss_fn = self.get_value_loss(batch)
    diff_loss_fn = self.get_diff_loss(batch)

    def policy_loss_fn(params, tgt_params, rng):
      observations = batch['observations']
      actions = batch['actions']

      rng, split_rng = jax.random.split(rng)
      # Calculate the guide loss
      diff_loss, terms, _, _ = diff_loss_fn(params, split_rng)
      action_dist = terms['action_dist']

      # Build action distribution
      replicated_obs = jnp.broadcast_to(
        observations, (self.config.sample_actions,) + observations.shape
      )
      rng, split_rng = jax.random.split(rng)
      if self.config.use_pred_astart:
        vf_actions = action_dist.sample(
          seed=split_rng, sample_shape=self.config.sample_actions
        )
      else:
        vf_actions = self.policy.apply(
          params['policy'], split_rng, replicated_obs
        )

      # Compute the current Q estimates
      cur_q1 = self.qf.apply(params['qf1'], observations, actions)
      cur_q2 = self.qf.apply(params['qf2'], observations, actions)

      # Compute values
      v1 = self.qf.apply(params['qf1'], replicated_obs, vf_actions)
      v2 = self.qf.apply(params['qf2'], replicated_obs, vf_actions)
      v = jnp.minimum(v1, v2)
      q_pred = jnp.minimum(cur_q1, cur_q2)
      avg_fn = getattr(jnp, self.config.crr_avg_fn)
      adv = q_pred - avg_fn(v, axis=0)
      if self.config.crr_fn == 'exp':
        lmbda = jnp.minimum(
          self.config.crr_ratio_upper_bound,
          jnp.exp(adv / self.config.crr_beta)
        )
        if self.config.adv_norm:
          lmbda = jax.nn.softmax(adv / self.config.crr_beta)
      else:
        lmbda = jnp.heaviside(adv, 0)
      lmbda = jax.lax.stop_gradient(lmbda)
      if self.config.crr_weight_mode == 'elbo':
        log_prob = -terms['ts_weights'] * terms['mse']
      elif self.config.crr_weight_mode == 'mle':
        log_prob = action_dist.log_prob(actions)
      else:
        rng, split_rng = jax.random.split(rng)
        if not self.config.crr_multi_sample_mse:
          sampled_actions = action_dist.sample(seed=split_rng)
          log_prob = -((sampled_actions - actions)**2).mean(axis=-1)
        else:
          sampled_actions = action_dist.sample(
            seed=split_rng, sample_shape=self.config.sample_actions
          )
          log_prob = -(
            (sampled_actions - jnp.expand_dims(actions, axis=0))**2
          ).mean(axis=(0, -1))
      guide_loss = -jnp.mean(log_prob * lmbda)

      policy_loss = self.config.diff_coef * diff_loss + \
        self.config.guide_coef * guide_loss
      losses = {'policy': policy_loss, 'policy_dist': policy_loss}
      return tuple(losses[key] for key in losses.keys()), locals()

    # Calculat policy losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_qf), grads_qf = value_and_multi_grad(
      value_loss_fn, 2, has_aux=True
    )(params, tgt_params, rng)

    (_, aux_policy), grads_policy = value_and_multi_grad(
      policy_loss_fn, 2, has_aux=True
    )(params, tgt_params, rng)

    # Update qf train states
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=grads_qf[0]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=grads_qf[1]['qf2']
    )

    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=grads_policy[0]['policy']
    )
    train_states['policy_dist'] = train_states['policy_dist'].apply_gradients(
      grads=grads_policy[1]['policy_dist']
    )

    # Update target parameters
    if policy_tgt_update:
      tgt_params['policy'] = update_target_network(
        train_states['policy'].params, tgt_params['policy'], self.config.tau
      )
    tgt_params['qf1'] = update_target_network(
      train_states['qf1'].params, tgt_params['qf1'], self.config.tau
    )
    tgt_params['qf2'] = update_target_network(
      train_states['qf2'].params, tgt_params['qf2'], self.config.tau
    )

    metrics = dict(
      qf_loss=aux_qf['qf_loss'],
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      cur_q1=aux_qf['cur_q1'].mean(),
      cur_q2=aux_qf['cur_q2'].mean(),
      tgt_q1=aux_qf['tgt_q1'].mean(),
      tgt_q2=aux_qf['tgt_q2'].mean(),
      tgt_q=aux_qf['tgt_q'].mean(),
      policy_loss=aux_policy['policy_loss'],
      guide_loss=aux_policy['guide_loss'],
      diff_loss=aux_policy['diff_loss'],
      lmbda=aux_policy['lmbda'].mean(),
      qf1_grad_norm=optax.global_norm(grads_qf[0]['qf1']),
      qf2_grad_norm=optax.global_norm(grads_qf[1]['qf2']),
      policy_grad_norm=optax.global_norm(grads_policy[0]['policy']),
      qf1_weight_norm=optax.global_norm(train_states['qf1'].params),
      qf2_weight_norm=optax.global_norm(train_states['qf2'].params),
      policy_weight_norm=optax.global_norm(train_states['policy'].params),
    )

    if self.config.loss_type == 'CRR':
      metrics['adv'] = aux_policy['adv'].mean()
      metrics['log_prob'] = aux_policy['log_prob'].mean()

    return train_states, tgt_params, metrics

  def _train_step_iql(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    diff_loss_fn = self.get_diff_loss(batch)

    def value_loss(train_params):
      observations = batch['observations']
      actions = batch['actions']
      q1 = self.qf.apply(tgt_params['qf1'], observations, actions)
      q2 = self.qf.apply(tgt_params['qf2'], observations, actions)
      q_pred = jax.lax.stop_gradient(jnp.minimum(q1, q2))
      v_pred = self.vf.apply(train_params['vf'], observations)
      diff = q_pred - v_pred
      expectile_weight = jnp.where(
        diff > 0,
        self.config.expectile,
        1 - self.config.expectile,
      )

      expectile_loss = (expectile_weight * (diff**2)).mean()
      return (expectile_loss,), locals()

    def critic_loss(train_params):
      observations = batch['observations']
      actions = batch['actions']
      next_observations = batch['next_observations']
      rewards = batch['rewards']
      dones = batch['dones']
      next_v = self.vf.apply(train_params['vf'], next_observations)

      discount = self.config.discount**self.config.nstep
      td_target = jax.lax.stop_gradient(
        rewards + (1 - dones) * discount * next_v
      )

      q1_pred = self.qf.apply(train_params['qf1'], observations, actions)
      q2_pred = self.qf.apply(train_params['qf2'], observations, actions)
      qf1_loss = mse_loss(q1_pred, td_target)
      qf2_loss = mse_loss(q2_pred, td_target)

      return (qf1_loss, qf2_loss), locals()

    def policy_loss(params, rng):
      observations = batch['observations']
      actions = batch['actions']
      rng, split_rng = jax.random.split(rng)
      diff_loss, terms, _, pred_astart = diff_loss_fn(params, split_rng)
      v_pred = self.vf.apply(train_params['vf'], observations)
      q1 = self.qf.apply(tgt_params['qf1'], observations, actions)
      q2 = self.qf.apply(tgt_params['qf2'], observations, actions)
      q_pred = jax.lax.stop_gradient(jnp.minimum(q1, q2))
      exp_a = jnp.exp((q_pred - v_pred) * self.config.awr_temperature)
      exp_a = jnp.minimum(exp_a, 100.0)

      if self.config.adv_norm:
        exp_a = jax.nn.softmax((q_pred - v_pred) * self.config.awr_temperature)

      action_dist = terms['action_dist']

      log_probs = action_dist.log_prob(actions)
      awr_loss = -(exp_a * log_probs).mean()
      guide_loss = awr_loss

      policy_loss = self.config.diff_coef * diff_loss + self.config.guide_coef * guide_loss
      losses = {'policy': policy_loss, 'policy_dist': policy_loss}

      return tuple(losses[key] for key in losses.keys()), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_value), value_grads = value_and_multi_grad(
      value_loss, 1, has_aux=True
    )(
      train_params
    )
    train_states['vf'] = train_states['vf'].apply_gradients(
      grads=value_grads[0]['vf']
    )

    rng, split_rng = jax.random.split(rng)
    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_policy), policy_grads = value_and_multi_grad(
      policy_loss, 2, has_aux=True
    )(train_params, split_rng)
    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=policy_grads[0]['policy']
    )
    train_states['policy_dist'] = train_states['policy_dist'].apply_gradients(
      grads=policy_grads[1]['policy_dist']
    )

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_qf), qf_grads = value_and_multi_grad(
      critic_loss, 2, has_aux=True
    )(
      train_params
    )
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=qf_grads[0]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=qf_grads[1]['qf2']
    )

    # Update target parameters
    if policy_tgt_update:
      tgt_params['policy'] = update_target_network(
        train_states['policy'].params, tgt_params['policy'], self.config.tau
      )
    tgt_params['qf1'] = update_target_network(
      train_states['qf1'].params, tgt_params['qf1'], self.config.tau
    )
    tgt_params['qf2'] = update_target_network(
      train_states['qf2'].params, tgt_params['qf2'], self.config.tau
    )

    metrics = dict(
      vf_loss=aux_value['expectile_loss'].mean(),
      vf_adv=aux_value['diff'].mean(),
      vf_pred=aux_value['v_pred'].mean(),
      next_v=aux_qf['next_v'].mean(),
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      cur_q1=aux_qf['q1_pred'].mean(),
      cur_q2=aux_qf['q2_pred'].mean(),
      tgt_q=aux_qf['td_target'].mean(),
      policy_loss=aux_policy['policy_loss'],
      guide_loss=aux_policy['guide_loss'],
      diff_loss=aux_policy['diff_loss'],
      vf_grad=optax.global_norm(value_grads[0]['vf']),
      qf1_grad_norm=optax.global_norm(qf_grads[0]['qf1']),
      qf2_grad_norm=optax.global_norm(qf_grads[1]['qf2']),
      policy_grad_norm=optax.global_norm(policy_grads[0]['policy']),
      vf_weight_norm=optax.global_norm(train_states['vf'].params),
      qf1_weight_norm=optax.global_norm(train_states['qf1'].params),
      qf2_weight_norm=optax.global_norm(train_states['qf2'].params),
      policy_weight_norm=optax.global_norm(train_states['policy'].params),
    )

    return train_states, tgt_params, metrics

  def train(self, batch):
    self._total_steps += 1
    policy_tgt_update = (
      self._total_steps > 1000 and
      self._total_steps % self.config.policy_tgt_freq == 0
    )
    self._train_states, self._tgt_params, metrics = self._train_step(
      self._train_states, self._tgt_params, next_rng(), batch,
      policy_tgt_update
    )
    return metrics

  @property
  def model_keys(self):
    return self._model_keys

  @property
  def train_states(self):
    return self._train_states

  @property
  def train_params(self):
    return {key: self.train_states[key].params for key in self.model_keys}

  @property
  def total_steps(self):
    return self._total_steps
