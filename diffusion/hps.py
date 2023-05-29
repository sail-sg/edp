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

hyperparameters = {
  'halfcheetah-medium-v2':
    {
      'lr': 3e-4,
      'eta': 1.0,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 2000,
      'gn': 9.0,
      'top_k': 1
    },
  'hopper-medium-v2':
    {
      'lr': 3e-4,
      'eta': 1.0,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 2000,
      'gn': 9.0,
      'top_k': 2
    },
  'walker2d-medium-v2':
    {
      'lr': 3e-4,
      'eta': 1.0,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 2000,
      'gn': 1.0,
      'top_k': 1
    },
  'halfcheetah-medium-replay-v2':
    {
      'lr': 3e-4,
      'eta': 1.0,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 2000,
      'gn': 2.0,
      'top_k': 0
    },
  'hopper-medium-replay-v2':
    {
      'lr': 3e-4,
      'eta': 1.0,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 2000,
      'gn': 4.0,
      'top_k': 2
    },
  'walker2d-medium-replay-v2':
    {
      'lr': 3e-4,
      'eta': 1.0,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 2000,
      'gn': 4.0,
      'top_k': 1
    },
  'halfcheetah-medium-expert-v2':
    {
      'lr': 3e-4,
      'eta': 1.0,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 2000,
      'gn': 7.0,
      'top_k': 0
    },
  'hopper-medium-expert-v2':
    {
      'lr': 3e-4,
      'eta': 1.0,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 2000,
      'gn': 5.0,
      'top_k': 2
    },
  'walker2d-medium-expert-v2':
    {
      'lr': 3e-4,
      'eta': 1.0,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 2000,
      'gn': 5.0,
      'top_k': 1
    },
  'antmaze-umaze-v0':
    {
      'lr': 3e-4,
      'eta': 0.5,
      'max_q_backup': False,
      'reward_tune': 'cql_antmaze',
      'eval_freq': 50,
      'num_epochs': 1000,
      'gn': 2.0,
      'top_k': 2
    },
  'antmaze-umaze-diverse-v0':
    {
      'lr': 3e-4,
      'eta': 2.0,
      'max_q_backup': True,
      'reward_tune': 'cql_antmaze',
      'eval_freq': 50,
      'num_epochs': 1000,
      'gn': 3.0,
      'top_k': 2
    },
  'antmaze-medium-play-v0':
    {
      'lr': 1e-3,
      'eta': 2.0,
      'max_q_backup': True,
      'reward_tune': 'cql_antmaze',
      'eval_freq': 50,
      'num_epochs': 1000,
      'gn': 2.0,
      'top_k': 1
    },
  'antmaze-medium-diverse-v0':
    {
      'lr': 3e-4,
      'eta': 3.0,
      'max_q_backup': True,
      'reward_tune': 'cql_antmaze',
      'eval_freq': 50,
      'num_epochs': 1000,
      'gn': 1.0,
      'top_k': 1
    },
  'antmaze-large-play-v0':
    {
      'lr': 3e-4,
      'eta': 4.5,
      'max_q_backup': True,
      'reward_tune': 'cql_antmaze',
      'eval_freq': 50,
      'num_epochs': 1000,
      'gn': 10.0,
      'top_k': 2
    },
  'antmaze-large-diverse-v0':
    {
      'lr': 3e-4,
      'eta': 3.5,
      'max_q_backup': True,
      'reward_tune': 'cql_antmaze',
      'eval_freq': 50,
      'num_epochs': 1000,
      'gn': 7.0,
      'top_k': 1
    },
  'pen-human-v1':
    {
      'lr': 3e-5,
      'eta': 0.15,
      'max_q_backup': False,
      'reward_tune': 'normalize',
      'eval_freq': 50,
      'num_epochs': 1000,
      'gn': 7.0,
      'top_k': 2
    },
  'pen-cloned-v1':
    {
      'lr': 3e-5,
      'eta': 0.1,
      'max_q_backup': False,
      'reward_tune': 'normalize',
      'eval_freq': 50,
      'num_epochs': 1000,
      'gn': 8.0,
      'top_k': 2
    },
  'kitchen-complete-v0':
    {
      'lr': 3e-4,
      'eta': 0.005,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 250,
      'gn': 9.0,
      'top_k': 2
    },
  'kitchen-partial-v0':
    {
      'lr': 3e-4,
      'eta': 0.005,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 1000,
      'gn': 10.0,
      'top_k': 2
    },
  'kitchen-mixed-v0':
    {
      'lr': 3e-4,
      'eta': 0.005,
      'max_q_backup': False,
      'reward_tune': 'no',
      'eval_freq': 50,
      'num_epochs': 1000,
      'gn': 10.0,
      'top_k': 0
    },
}
