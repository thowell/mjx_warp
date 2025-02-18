"""Tests for solver functions."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import mujoco
from . import io
from . import smooth
from . import solver
import numpy as np
import warp as wp

# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-3


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SolverTest(parameterized.TestCase):
  def _load(
    self,
    fname: str,
    is_sparse: bool = True,
    cone: int = mujoco.mjtCone.mjCONE_PYRAMIDAL,
    solver_: int = mujoco.mjtSolver.mjSOL_NEWTON,
    iterations: int = 2,
  ):
    path = epath.resource_path("mujoco.mjx") / "test_data" / fname
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
    mjm.opt.jacobian = is_sparse
    mjm.opt.iterations = iterations
    mjm.opt.cone = cone
    mjm.opt.solver = solver_
    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_step(mjm, mjd)
    m = io.put_model(mjm)
    d = io.put_data(mjm, mjd)
    return mjm, mjd, m, d

  @parameterized.parameters(
    (mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtSolver.mjSOL_CG, 100),
    (mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtSolver.mjSOL_NEWTON, 2),
  )
  def test_solve_pyramidal(self, cone, solver_, iterations):
    """Tests MJX solve."""
    mjm, mjd, m, d = self._load(
      "humanoid/humanoid.xml",
      is_sparse=False,
      cone=cone,
      solver_=solver_,
      iterations=iterations,
    )

    def cost(qacc):
      jaref = np.zeros(mjd.nefc, dtype=float)
      cost = np.zeros(1)
      mujoco.mj_mulJacVec(mjm, mjd, jaref, qacc)
      mujoco.mj_constraintUpdate(mjm, mjd, jaref - mjd.efc_aref, cost, 0)
      return cost

    mj_cost = cost(mjd.qacc)

    ctx = solver._context(m, d)
    solver._create_context(ctx, m, d)

    mjx_cost = ctx.cost.numpy()[0] - ctx.gauss.numpy()[0]

    _assert_eq(mjx_cost, mj_cost, name="cost")

    qacc_warmstart = mjd.qacc_warmstart.copy()
    mujoco.mj_forward(mjm, mjd)
    mjd.qacc_warmstart = qacc_warmstart

    m = io.put_model(mjm)
    d = io.put_data(mjm, mjd)
    d.qacc.zero_()
    d.qfrc_constraint.zero_()
    d.efc_force.zero_()

    smooth.factor_m(m, d)  # if dense get tile cholesky factor
    solver.solve(m, d)

    mj_cost = cost(mjd.qacc)
    mjx_cost = cost(d.qacc.numpy()[0])
    self.assertLess(mjx_cost, mj_cost * 1.015)

    if m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON:
      _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
      _assert_eq(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, "qfrc_constraint")
      _assert_eq(d.efc_force.numpy()[0], mjd.efc_force, "efc_force")


if __name__ == "__main__":
  wp.init()
  absltest.main()
