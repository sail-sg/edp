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

"""General utils for training."""

import os
import pprint
import random
import tempfile
import time
import uuid
from copy import copy
from socket import gethostname

import absl.flags
import cloudpickle as pickle
import numpy as np
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags

from utilities.jax_utils import init_rng


def norm_obs(ds, mean, std, clip_val):
  ds["observations"] = (ds["observations"] - mean) / (std + 1e-6)
  ds["next_observations"] = (ds["next_observations"] - mean) / (std + 1e-6)

  ds["observations"] = np.clip(ds["observations"], -clip_val, clip_val)
  ds["next_observations"] = np.clip(
    ds["next_observations"], -clip_val, clip_val
  )


class Timer(object):

  def __init__(self):
    self._time = None

  def __enter__(self):
    self._start_time = time.time()
    return self

  def __exit__(self, exc_type, exc_value, exc_tb):
    self._time = time.time() - self._start_time

  def __call__(self):
    return self._time


class WandBLogger(object):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.team = 'jax_offrl'
    config.online = False
    config.prefix = ""
    config.project = "OfflineRL"
    config.output_dir = "/tmp/diffusion_rl"
    config.random_delay = 0.0
    config.experiment_id = config_dict.placeholder(str)
    config.anonymous = config_dict.placeholder(str)
    config.notes = config_dict.placeholder(str)

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, variant, env_name):
    self.config = self.get_default_config(config)

    if self.config.experiment_id is None:
      self.config.experiment_id = uuid.uuid4().hex

    if self.config.prefix != "":
      self.config.project = "{}--{}".format(
        self.config.prefix, self.config.project
      )

    if self.config.output_dir == "":
      self.config.output_dir = tempfile.mkdtemp()
    else:
      self.config.output_dir = os.path.join(
        self.config.output_dir, self.config.experiment_id
      )
      os.makedirs(self.config.output_dir, exist_ok=True)

    self._variant = copy(variant)

    if "hostname" not in self._variant:
      self._variant["hostname"] = gethostname()

    if self.config.random_delay > 0:
      time.sleep(np.random.uniform(0, self.config.random_delay))

    self.run = wandb.init(
      entity=self.config.team,
      reinit=True,
      config=self._variant,
      project=self.config.project,
      dir=self.config.output_dir,
      id=self.config.experiment_id,
      anonymous=self.config.anonymous,
      notes=self.config.notes,
      settings=wandb.Settings(
        start_method="thread",
        _disable_stats=True,
      ),
      mode="online" if self.config.online else "offline",
    )

  def log(self, *args, **kwargs):
    self.run.log(*args, **kwargs)

  def save_pickle(self, obj, filename):
    with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
      pickle.dump(obj, fout)

  @property
  def experiment_id(self):
    return self.config.experiment_id

  @property
  def variant(self):
    return self.config.variant

  @property
  def output_dir(self):
    return self.config.output_dir


def define_flags_with_default(**kwargs):
  for key, val in kwargs.items():
    if isinstance(val, ConfigDict):
      config_flags.DEFINE_config_dict(key, val)
    elif isinstance(val, bool):
      # Note that True and False are instances of int.
      absl.flags.DEFINE_bool(key, val, "automatically defined flag")
    elif isinstance(val, int):
      absl.flags.DEFINE_integer(key, val, "automatically defined flag")
    elif isinstance(val, float):
      absl.flags.DEFINE_float(key, val, "automatically defined flag")
    elif isinstance(val, str):
      absl.flags.DEFINE_string(key, val, "automatically defined flag")
    else:
      raise ValueError("Incorrect value type")
  return kwargs


def set_random_seed(seed):
  np.random.seed(seed)
  random.seed(seed)
  init_rng(seed)


def print_flags(flags, flags_def):
  logging.info(
    "Running training with hyperparameters: \n{}".format(
      pprint.pformat(
        [
          "{}: {}".format(key, val)
          for key, val in get_user_flags(flags, flags_def).items()
        ]
      )
    )
  )


def get_user_flags(flags, flags_def):
  output = {}
  for key in flags_def:
    val = getattr(flags, key)
    if isinstance(val, ConfigDict):
      output.update(flatten_config_dict(val, prefix=key))
    else:
      output[key] = val

  return output


def flatten_config_dict(config, prefix=None):
  output = {}
  for key, val in config.items():
    if isinstance(val, ConfigDict):
      output.update(flatten_config_dict(val, prefix=key))
    else:
      if prefix is not None:
        output["{}.{}".format(prefix, key)] = val
      else:
        output[key] = val
  return output


def prefix_metrics(metrics, prefix):
  return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}
