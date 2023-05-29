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

"""Dataset samplers."""
from typing import Any

import numpy as np

Array = Any


class RandSampler(object):
	"""A random sampler."""

	def __init__(self, max_size: int, batch_size: int = 1) -> None:
		self._max_size = max_size
		self._batch_size = batch_size

	def sample(self):
		"""Return an array of sampled indices."""

		return np.random.randint(self._max_size, size=self._batch_size)
