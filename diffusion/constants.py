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

from enum import IntEnum


class ENV(IntEnum):
  Adroit = 1
  Kitchen = 2
  Mujoco = 3
  Antmaze = 4


ENV_MAP = {
  'pen': ENV.Adroit,
  'hammer': ENV.Adroit,
  'door': ENV.Adroit,
  'relocate': ENV.Adroit,
  'kitchen': ENV.Kitchen,
  'hopper': ENV.Mujoco,
  'walker': ENV.Mujoco,
  'cheetah': ENV.Mujoco,
  'finger': ENV.Mujoco,
  'humanoid': ENV.Mujoco,
  'cartpole': ENV.Mujoco,
  'fish': ENV.Mujoco,
  'antmaze': ENV.Antmaze
}

ENVNAME_MAP = {
  ENV.Adroit: 'Adroit',
  ENV.Kitchen: 'Kitchen',
  ENV.Mujoco: 'Mujoco',
  ENV.Antmaze: 'Antmaze'
}


class DATASET(IntEnum):
  D4RL = 1
  RLUP = 2


DATASET_MAP = {'d4rl': DATASET.D4RL, 'rl_unplugged': DATASET.RLUP}

DATASET_ABBR_MAP = {'d4rl': 'D4RL', 'rl_unplugged': 'RLUP'}
