"""Tests for passive force functions."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import mujoco
from mujoco import mjx
import numpy as np
import warp as wp

# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5

def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class PassiveTest(parameterized.TestCase):

  def _load(self, fname: str, is_sparse: bool = True):
    path = epath.resource_path('mujoco.mjx') / 'test_data' / fname
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
    mjm.opt.jacobian = is_sparse
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 1) # reset to stand_on_left_leg
    mjd.qvel = np.random.uniform(low=-0.01, high=0.01, size=mjd.qvel.shape)
    mujoco.mj_forward(mjm, mjd)
    m = mjx.put_model(mjm)
    d = mjx.put_data(mjm, mjd)
    return mjm, mjd, m, d

  @parameterized.parameters('humanoid/humanoid.xml', 'humanoid/n_humanoids.xml')
  def test_passive(self, fname: str):
    """Tests MJX passive."""
    _, mjd, m, d = self._load(fname)

    for arr in (d.qfrc_passive, d.qfrc_spring, d.qfrc_damper):
      arr.zero_()

    mjx.passive(m, d)

    _assert_eq(d.qfrc_passive.numpy()[0], mjd.qfrc_passive, 'qfrc_passive')
    _assert_eq(d.qfrc_spring.numpy()[0], mjd.qfrc_spring, 'qfrc_spring')
    _assert_eq(d.qfrc_damper.numpy()[0], mjd.qfrc_damper, 'qfrc_damper')


if __name__ == '__main__':
  wp.init()
  absltest.main()
