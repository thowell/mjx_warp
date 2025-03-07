# Copyright 2025 The Physics-Next Project Developers
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
# ==============================================================================

from typing import Callable, Optional

import warp as wp
from warp.context import Module, get_module


def kernel(
  f: Optional[Callable] = None,
  *,
  enable_backward: Optional[bool] = None,
  module: Optional[Module] = None,
):
  if module is None:
    # create a module name based on the name of the nested function
    # get the qualified name, e.g. "main.<locals>.nested_kernel"
    qualname = f.__qualname__
    parts = [part for part in qualname.split(".") if part != "<locals>"]
    outer_functions = parts[:-1]
    module = get_module(".".join([f.__module__] + outer_functions))

  return wp.kernel(f, enable_backward=enable_backward, module=module)
