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
    mx = mjx.put_model(m)
    dx = mjx.make_data(m)
    qpos = np.reshape(d.qpos, (1, -1))
    dx.qpos = wp.array(qpos, dtype=wp.float32, ndim=2)

    mjx.kinematics(mx, dx)
    _assert_eq(d.xanchor, dx.xanchor.numpy().reshape((-1, 3)), 'xanchor')
    _assert_eq(d.xaxis, dx.xaxis.numpy().reshape((-1, 3)), 'xaxis')
    _assert_eq(d.xquat, dx.xquat.numpy().reshape((-1, 4)), 'xquat')
    _assert_eq(d.xpos, dx.xpos.numpy().reshape((-1, 3)), 'xpos')

  def test_com_pos(self):
    """Tests MJX com_pos."""
    path = epath.resource_path('mujoco.mjx') / 'test_data/humanoid/humanoid.xml'
    m = mujoco.MjModel.from_xml_path(path.as_posix())
    d = mujoco.MjData(m)
    # reset to stand_on_left_leg
    mujoco.mj_resetDataKeyframe(m, d, 1)
    mujoco.mj_forward(m, d)
    # make mjx model and data
    mx = mjx.put_model(m)
    dx = mjx.make_data(m)
    qpos = np.reshape(d.qpos, (1, -1))
    dx.qpos = wp.array(qpos, dtype=wp.float32, ndim=2)
    xaxis = np.reshape(d.xaxis, (1, -1, 3))
    dx.xaxis = wp.array(xaxis, dtype=wp.vec3, ndim=2)
    xanchor = np.reshape(d.xanchor, (1, -1, 3))
    dx.xanchor = wp.array(xanchor, dtype=wp.vec3, ndim=2)
    xmat = np.reshape(d.xmat, (1, -1, 3, 3))
    dx.xmat = wp.array(xmat, dtype=wp.mat33, ndim=2)
    xipos = np.reshape(d.xipos, (1, -1, 3))
    dx.xipos = wp.array(xipos, dtype=wp.vec3, ndim=2)
    ximat = np.reshape(d.ximat, (1, -1, 3, 3))
    dx.ximat = wp.array(ximat, dtype=wp.mat33, ndim=2)

    mjx.com_pos(mx, dx)
    _assert_eq(dx.subtree_com.numpy().reshape((-1, 3)), d.subtree_com, 'subtree_com')
    _assert_eq(dx.cinert.numpy().reshape((-1, 10)), d.cinert, 'subtree_com')
    np.set_printoptions(suppress=True, linewidth=1000, precision=4)
    print(dx.cdof.numpy().reshape((-1, 6)))
    print(d.cdof)
    _assert_eq(dx.cdof.numpy().reshape((-1, 6)), d.cdof, 'subtree_com')


if __name__ == '__main__':
  wp.init()
  absltest.main()
