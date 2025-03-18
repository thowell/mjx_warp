# Copyright 2025 The Newton Developers
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

"""Run benchmarks on various devices."""

import inspect
from typing import Sequence

import mujoco
import numpy as np
import warp as wp
from absl import app
from absl import flags
from etils import epath

import mujoco_warp as mjwarp

_FUNCTION = flags.DEFINE_enum(
  "function",
  "step",
  [n for n, _ in inspect.getmembers(mjwarp, inspect.isfunction)],
  "the function to run",
)
_MJCF = flags.DEFINE_string(
  "mjcf", None, "path to model `.xml` or `.mjb`", required=True
)
_BASE_PATH = flags.DEFINE_string(
  "base_path", None, "base path, defaults to mujoco_warp resource path"
)
_NSTEP = flags.DEFINE_integer("nstep", 1000, "number of steps per rollout")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 4096, "number of parallel rollouts")
_SOLVER = flags.DEFINE_enum("solver", "newton", ["cg", "newton"], "constraint solver")
_ITERATIONS = flags.DEFINE_integer("iterations", 1, "number of solver iterations")
_LS_ITERATIONS = flags.DEFINE_integer(
  "ls_iterations", 4, "number of linesearch iterations"
)
_LS_PARALLEL = flags.DEFINE_bool("ls_parallel", False, "solve with parallel linesearch")
_IS_SPARSE = flags.DEFINE_bool(
  "is_sparse", False, "if model should create sparse mass matrices"
)
_NCONMAX = flags.DEFINE_integer(
  "nconmax", -1, "Maximum number of contacts in a batch physics step."
)
_NJMAX = flags.DEFINE_integer(
  "njmax", -1, "Maximum number of constraints in a batch physics step."
)
_KEYFRAME = flags.DEFINE_integer("keyframe", 0, "Keyframe to initialize simulation.")
_OUTPUT = flags.DEFINE_enum(
  "output", "text", ["text", "tsv"], "format to print results"
)
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool(
  "clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)"
)
_EVENT_TRACE = flags.DEFINE_bool("event_trace", False, "Provide a full event trace")
_MEASURE_ALLOC = flags.DEFINE_bool(
  "measure_alloc", False, "Measure how much of nconmax, njmax is used."
)


def _main(argv: Sequence[str]):
  """Runs testpeed function."""
  wp.init()

  path = epath.resource_path("mujoco_warp") / "test_data"
  path = _BASE_PATH.value or path
  f = epath.Path(path) / _MJCF.value
  if f.suffix == ".mjb":
    m = mujoco.MjModel.from_binary_path(f.as_posix())
  else:
    m = mujoco.MjModel.from_xml_path(f.as_posix())

  if _IS_SPARSE.value:
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  else:
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

  d = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d, _KEYFRAME.value)
  # populate some constraints
  mujoco.mj_forward(m, d)

  if _CLEAR_KERNEL_CACHE.value:
    wp.clear_kernel_cache()

  print(
    f"Model nbody: {m.nbody} nv: {m.nv} ngeom: {m.ngeom} is_sparse: {_IS_SPARSE.value} solver: {_SOLVER.value}"
  )
  print(f"Params nconmax: {_NCONMAX.value} njmax: {_NJMAX.value}")
  print(f"Data ncon: {d.ncon} nefc: {d.nefc} keyframe: {_KEYFRAME.value}")
  print(f"Linesearch: {'parallel' if _LS_PARALLEL.value else 'iterative'}")
  print(f"Rolling out {_NSTEP.value} steps at dt = {m.opt.timestep:.3f}...")
  jit_time, run_time, trace, steps, ncon, nefc = mjwarp.benchmark(
    mjwarp.__dict__[_FUNCTION.value],
    m,
    d,
    _NSTEP.value,
    _BATCH_SIZE.value,
    _SOLVER.value,
    _ITERATIONS.value,
    _LS_ITERATIONS.value,
    _LS_PARALLEL.value,
    _NCONMAX.value,
    _NJMAX.value,
    _EVENT_TRACE.value,
    _MEASURE_ALLOC.value,
  )

  name = argv[0]
  if _OUTPUT.value == "text":
    print(f"""
Summary for {_BATCH_SIZE.value} parallel rollouts

 Total JIT time: {jit_time:.2f} s
 Total simulation time: {run_time:.2f} s
 Total steps per second: {steps / run_time:,.0f}
 Total realtime factor: {steps * m.opt.timestep / run_time:,.2f} x
 Total time per step: {1e9 * run_time / steps:.2f} ns""")
    if trace:
      print("\nEvent trace:\n")

      def _print_trace(trace, indent):
        for k, v in trace.items():
          times, sub_trace = v
          if len(times) == 1:
            print("  " * indent + f"{k}: {1e6 * times[0] / steps:.2f}")
          else:
            print("  " * indent + f"{k}: [ ", end="")
            for i in range(len(times)):
              print(f"{1e6 * times[i] / steps:.2f}", end="")
              print(", " if i < len(times) - 1 else " ", end="")
            print("]")
          _print_trace(sub_trace, indent + 1)

      _print_trace(trace, 0)
    if ncon and nefc:
      num_buckets = 10
      idx = 0
      ncon_matrix, nefc_matrix = [], []
      for i in range(num_buckets):
        size = _NSTEP.value // num_buckets + (i < (_NSTEP.value % num_buckets))
        ncon_arr = np.array(ncon[idx : idx + size])
        nefc_arr = np.array(nefc[idx : idx + size])
        ncon_matrix.append(
          [np.mean(ncon_arr), np.std(ncon_arr), np.min(ncon_arr), np.max(ncon_arr)]
        )
        nefc_matrix.append(
          [np.mean(nefc_arr), np.std(nefc_arr), np.min(nefc_arr), np.max(nefc_arr)]
        )
        idx += size

      def _print_table(matrix, headers):
        num_cols = len(headers)
        col_widths = [
          max(len(f"{row[i]:g}") for row in matrix) for i in range(num_cols)
        ]
        col_widths = [max(col_widths[i], len(headers[i])) for i in range(num_cols)]

        print("  ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(num_cols)))
        print("-" * sum(col_widths) + "--" * 3)  # Separator line
        for row in matrix:
          print("  ".join(f"{row[i]:{col_widths[i]}g}" for i in range(num_cols)))

      print("\nncon alloc:\n")
      _print_table(ncon_matrix, ("mean", "std", "min", "max"))
      print("\nnefc alloc:\n")
      _print_table(nefc_matrix, ("mean", "std", "min", "max"))

  elif _OUTPUT.value == "tsv":
    name = name.split("/")[-1].replace("testspeed_", "")
    print(f"{name}\tjit: {jit_time:.2f}s\tsteps/second: {steps / run_time:.0f}")


def main():
  app.run(_main)


if __name__ == "__main__":
  main()
