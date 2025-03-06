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

import functools
import inspect
from typing import Callable, Optional

from warp.codegen import make_full_qualified_name
from warp.context import Function, Kernel, Module, get_module

USE_SEPARATE_MODULES = True


def func(
  f: Optional[Callable] = None,
  *,
  name: Optional[str] = None,
  module: Optional[Module] = None,
):
  """
  Extends the @wp.func decorator to accept an extra `module` parameter.
  If `module` is not provided, it falls back to `get_module(f.__module__)`.
  """

  def wrapper(func_to_wrap: Callable, *args, **kwargs):
    # Determine key (either user-provided `name` or derived from function).
    if name is None:
      key = make_full_qualified_name(func_to_wrap)  # or your codegen call
    else:
      key = name

    # We do a little frame inspection (from your existing code)
    scope_locals = inspect.currentframe().f_back.f_back.f_locals

    # If user provided a module, use it. Otherwise use get_module from f's __module__.
    m = module or get_module(func_to_wrap.__module__)

    doc = getattr(func_to_wrap, "__doc__", "") or ""

    # Pass the `module` that was explicitly provided or the fallback `m`.
    Function(
      func=func_to_wrap,
      key=key,
      namespace="",
      module=m,
      value_func=None,
      scope_locals=scope_locals,
      doc=doc.strip(),
    )

    # use the top of the list of overloads for this key
    g = m.functions[key]

    # copy over function attributes, including docstring
    return functools.update_wrapper(g, func_to_wrap)

  # If the decorator was called with arguments, return the real decorator (wrapper).
  # Otherwise (if it was just @func with no parentheses), call `wrapper` immediately.
  if f is None:
    return wrapper
  else:
    return wrapper(f)


def kernel(
  f: Optional[Callable] = None,
  *,
  enable_backward: Optional[bool] = None,
  module: Optional[Module] = None,
):
  def wrapper(f, *args, **kwargs):
    options = {}

    if enable_backward is not None:
      options["enable_backward"] = enable_backward

    m = module or get_module(f.__module__)
    k = Kernel(
      func=f,
      key=make_full_qualified_name(f),
      module=m,
      options=options,
    )
    k = functools.update_wrapper(k, f)
    return k

  if f is None:
    # Arguments were passed to the decorator.
    return wrapper

  return wrapper(f)
