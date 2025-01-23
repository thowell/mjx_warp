"""Tests for smooth dynamics functions."""

from absl.testing import absltest
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


class SmoothTest(absltest.TestCase):

  def test_kinematics(self):
    """Tests MJX kinematics."""
    path = epath.resource_path('mujoco.mjx') / 'test_data/humanoid/humanoid.xml'
    m = mujoco.MjModel.from_xml_path(path.as_posix())
    d = mujoco.MjData(m)
    # reset to stand_on_left_leg
    mujoco.mj_resetDataKeyframe(m, d, 1)
    mujoco.mj_forward(m, d)
    # make mjx model and data
    wp.init()
    mx = mjx.put_model(m)
    dx = mjx.make_data(m)
    qpos = np.tile(d.qpos, (1, 1))
    dx.qpos = wp.array(qpos, dtype=wp.float32, ndim=2)

    # common code path: only care about xpos / xquat
    mjx.kinematics(mx, dx, emit_anchor_axis=False)
    _assert_eq(d.xquat, dx.xquat.numpy().reshape((-1, 4)), 'xquat')
    _assert_eq(d.xpos, dx.xpos.numpy().reshape((-1, 3)), 'xpos')

    # for debugging you might want to see xanchor and xaxis
    mjx.kinematics(mx, dx, emit_anchor_axis=True)
    _assert_eq(d.xanchor, dx.xanchor.numpy().reshape((-1, 3)), 'xanchor')
    _assert_eq(d.xaxis, dx.xaxis.numpy().reshape((-1, 3)), 'xaxis')
    _assert_eq(d.xquat, dx.xquat.numpy().reshape((-1, 4)), 'xquat')
    _assert_eq(d.xpos, dx.xpos.numpy().reshape((-1, 3)), 'xpos')


if __name__ == '__main__':
  absltest.main()
