import warp as wp
from . import math
from . import types

@wp.kernel
def _kinematics_root(m: types.Model, d: types.Data):
  worldid = wp.tid()
  d.xpos[worldid, 0] = wp.vec3(0.0)
  d.xquat[worldid, 0] = wp.quat(1.0, 0.0, 0.0, 0.0)
  d.xipos[worldid, 0] = wp.vec3(0.0)
  d.xmat[worldid, 0] = wp.identity(n=3, dtype=wp.float32)
  d.ximat[worldid, 0] = wp.identity(n=3, dtype=wp.float32)

@wp.kernel
def _kinematics_level(m: types.Model, d: types.Data, level: int):
  worldid, levelid = wp.tid()
  bodyid = m.body_bfs[m.level_beg[level] + levelid]
  jntadr = m.body_jntadr[bodyid]
  jntnum = m.body_jntnum[bodyid]
  qpos = d.qpos[worldid]

  if jntnum == 1 and m.jnt_type[jntadr] == 0:
    # free joint
    qadr = m.jnt_qposadr[jntadr]
    # TODO(erikfrey): would it be better to use some kind of wp.copy here?
    xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])
    xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])
    d.xanchor[worldid, jntadr] = xpos
    d.xaxis[worldid, jntadr] = m.jnt_axis[jntadr]
  else:
    # regular or no joints
    # apply fixed translation and rotation relative to parent
    pid = m.body_parentid[bodyid]
    xpos = (d.xmat[worldid, pid] * m.body_pos[bodyid]) + d.xpos[worldid, pid]
    xquat = math.mul_quat(d.xquat[worldid, pid], m.body_quat[bodyid])

    for _ in range(jntnum):
      qadr = m.jnt_qposadr[jntadr]
      jnt_type = m.jnt_type[jntadr]
      xanchor = math.rot_vec_quat(m.jnt_pos[jntadr], xquat) + xpos
      xaxis = math.rot_vec_quat(m.jnt_axis[jntadr], xquat)

      if jnt_type == 3:  # hinge
        qloc = math.axis_angle_to_quat(m.jnt_axis[jntadr], d.qpos[worldid, qadr] - m.qpos0[qadr])
        xquat = math.mul_quat(xquat, qloc)
        # correct for off-center rotation
        xpos = xanchor - math.rot_vec_quat(m.jnt_pos[jntadr], xquat)

      d.xanchor[worldid, jntadr] = xanchor
      d.xaxis[worldid, jntadr] = xaxis
      jntadr += 1

  d.xpos[worldid, bodyid] = xpos
  d.xquat[worldid, bodyid] = wp.normalize(xquat)
  d.xmat[worldid, bodyid] = math.quat_to_mat(xquat)


def kinematics(m: types.Model, d: types.Data):
  wp.launch(_kinematics_root, dim=(d.nworld), inputs=[m, d])
  level_beg, level_end = m.level_beg_cpu.numpy(), m.level_end_cpu.numpy()
  for i in range(m.nlevel):
    wp.launch(_kinematics_level, dim=(d.nworld, level_end[i] - level_beg[i]), inputs=[m, d, i])
