import warp as wp
from . import math
from . import types


def passive(m: types.Model, d: types.Data):
  """Adds all passive forces."""

  @wp.kernel
  def _spring(m: types.Model, d: types.Data):
    worldid, jntid = wp.tid()
    stiffness = m.jnt_stiffness[jntid]

    if stiffness == 0.0:
      return

    jnt_type = m.jnt_type[jntid]
    padr, dadr = m.jnt_qposadr[jntid], m.jnt_dofadr[jntid]

    if jnt_type == 0:  # mjJNT_FREE
      dif = d.qpos[worldid, padr + 0] - m.qpos_spring[padr + 0]
      d.qfrc_spring[worldid, dadr + 0] = -stiffness * dif
      dif = d.qpos[worldid, padr + 1] - m.qpos_spring[padr + 1]
      d.qfrc_spring[worldid, dadr + 1] = -stiffness * dif
      dif = d.qpos[worldid, padr + 2] - m.qpos_spring[padr + 2]
      d.qfrc_spring[worldid, dadr + 2] = -stiffness * dif
      rot = wp.quat(
        d.qpos[worldid, padr + 3],
        d.qpos[worldid, padr + 4],
        d.qpos[worldid, padr + 5],
        d.qpos[worldid, padr + 6],
      )
      ref = wp.quat(
        m.qpos_spring[padr + 3],
        m.qpos_spring[padr + 4],
        m.qpos_spring[padr + 5],
        m.qpos_spring[padr + 6],
      )
      qdif = math.quat_sub(rot, ref)
      d.qfrc_spring[worldid, dadr + 3] = -stiffness * qdif[0]
      d.qfrc_spring[worldid, dadr + 4] = -stiffness * qdif[1]
      d.qfrc_spring[worldid, dadr + 5] = -stiffness * qdif[2]
    elif jnt_type == 1:  # mjJNT_BALL
      rot = wp.quat(
        d.qpos[worldid, padr + 0],
        d.qpos[worldid, padr + 1],
        d.qpos[worldid, padr + 2],
        d.qpos[worldid, padr + 3],
      )
      ref = wp.quat(
        m.qpos_spring[padr + 0],
        m.qpos_spring[padr + 1],
        m.qpos_spring[padr + 2],
        m.qpos_spring[padr + 3],
      )
      qdif = math.quat_sub(rot, ref)
      d.qfrc_spring[worldid, dadr + 0] = -stiffness * qdif[0]
      d.qfrc_spring[worldid, dadr + 1] = -stiffness * qdif[1]
      d.qfrc_spring[worldid, dadr + 2] = -stiffness * qdif[2]
    elif jnt_type == 2 or jnt_type == 3:  # mjJNT_SLIDE, mjJNT_HINGE
      dif = d.qpos[worldid, padr] - m.qpos_spring[padr]
      d.qfrc_spring[worldid, dadr] = -stiffness * dif

  @wp.kernel
  def _damper_passive(m: types.Model, d: types.Data):
    worldid, dofid = wp.tid()
    damping = m.dof_damping[dofid]

    d.qfrc_damper[worldid, dofid] = -damping * d.qvel[worldid, dofid]
    d.qfrc_passive[worldid, dofid] = d.qfrc_damper[worldid, dofid] + d.qfrc_spring[worldid, dofid]

  d.qfrc_spring.zero_()
  wp.launch(_spring, dim=(d.nworld, m.njnt), inputs=[m, d])
  wp.launch(_damper_passive, dim=(d.nworld, m.nv), inputs=[m, d])
