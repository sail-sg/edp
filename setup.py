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

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))


def get_version() -> str:
  # https://packaging.python.org/guides/single-sourcing-package-version/
  init = open(os.path.join("algos", "__init__.py"), "r").read().split()
  return init[init.index("__version__") + 2][1:-1]


with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
  requirements = f.read().splitlines()

setup(
  name="offrl",
  version=get_version(),
  description="Offline RL with Jax",
  long_description=open("README.md", encoding="utf8").read(),
  long_description_content_type="text/markdown",
  author="Jax OffRL Contributors",
  license="Apache 2.0",
  python_requires=">=3.6",
  classifiers=[
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 3 - Alpha",
    # Indicate who your project is intended for
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    # Pick your license as you wish (should match "license" above)
    "License :: OSI Approved :: Apache Software License",
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
  ],
  keywords="jax offline reinforcement learning",
  packages=find_packages(exclude=[]),
  install_requires=requirements,
)
