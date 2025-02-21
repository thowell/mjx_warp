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

import warp as wp
from . import math
from . import types
from typing import Any


@wp.struct
class _Efc:
  worldid: int
  J: wp.array(ndim=1, dtype=wp.float32)
  pos_aref: wp.array(ndim=1, dtype=wp.float32)
  pos_imp: wp.array(ndim=1, dtype=wp.float32)
  invweight: wp.array(ndim=1, dtype=wp.float32)
  solref: wp.array(ndim=1, dtype=wp.float32)
  solimp: wp.array(ndim=1, dtype=wp.float32)
  margin: wp.array(ndim=1, dtype=wp.float32)
  frictionloss: wp.array(ndim=1, dtype=wp.float32)


@wp.kernel
def _update_contact_data(
  m: types.Model,
  d: types.Data,
  efcs: wp.array(dtype=_Efc),
  refsafe: bool
):
  id = wp.tid()
  efc = efcs[id]
  worldid = efc.worldid

  # Calculate kbi
  timeconst = efc.solref[0]
  dampratio = efc.solref[1]
  dmin = efc.solimp[0]
  dmax = efc.solimp[1]
  width = efc.solimp[2]
  mid = efc.solimp[3]
  power = efc.solimp[4]

  if refsafe:
    timeconst = wp.max(timeconst, 2.0 * m.opt.timestep)

  dmin = wp.clamp(dmin, types.MINIMP, types.MAXIMP)
  dmax = wp.clamp(dmax, types.MINIMP, types.MAXIMP)
  width = wp.max(types.MINVAL, width)
  mid = wp.clamp(mid, types.MINIMP, types.MAXIMP)
  power = wp.max(1.0, power)

  # See https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
  k = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)
  b = 2.0 / (dmax * timeconst)
  # TODO(robotics-simulation): check various solparam settings in model gen test
  k = math.where(efc.solref[0] <= 0, -efc.solref[0] / (dmax * dmax), k)
  b = math.where(efc.solref[1] <= 0, -efc.solref[1] / dmax, b)

  imp_x = wp.abs(efc.pos_imp[0]) / width
  imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
  imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)
  imp_y = math.where(imp_x < mid, imp_a, imp_b)
  imp = dmin + imp_y * (dmax - dmin)
  imp = wp.clamp(imp, dmin, dmax)
  imp = math.where(imp_x > 1.0, dmax, imp)

  # Update constraints
  r = wp.max(efc.invweight[0] * (1.0 - imp) / imp, types.MINVAL)
  aref = float(0.0)
  for i in range(m.nv):
    aref += -b * (efc.J[i] * d.qvel[worldid, i])
    d.efc_J[worldid, id, i] = efc.J[i]
  aref -= k * imp * efc.pos_aref[0]
  d.efc_D[worldid, id] = 1.0 / r
  d.efc_aref[worldid, id] = aref
  d.efc_pos[worldid, id]  = efc.pos_aref[0] + efc.margin[0]
  d.efc_margin[worldid, id]  = efc.margin[0]
  d.efc_frictionloss[worldid, id]  = efc.frictionloss[0]

@wp.func
def _jac(m: types.Model, d: types.Data, worldid: wp.int32, point: wp.vec3, xyz: wp.int32, bodyid: wp.int32, dofid: wp.int32):
  dof_bodyid = m.dof_bodyid[dofid]
  in_tree = int(dof_bodyid == 0)
  parentid = bodyid
  while parentid != 0:
    if (parentid == dof_bodyid):
      in_tree = 1
      break
    parentid = m.body_parentid[parentid]

  offset = point - wp.vec3(d.subtree_com[worldid, m.body_rootid[bodyid]])
 
  temp_jac = wp.vec3(0.0)
  temp_jac = wp.spatial_bottom(d.cdof[worldid, dofid]) + wp.cross(wp.spatial_top(d.cdof[worldid, dofid]), offset)
  jacp = temp_jac[xyz] * float(in_tree)
  jacr = d.cdof[worldid, dofid][xyz] * float(in_tree)

  return jacp, jacr


@wp.kernel
def _efc_equality_connect(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), eq_id: wp.array(dtype=wp.int32), equality_connect: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  c_id = id // 3
  xyz_id = id % 3
  n_id = eq_id[c_id]

  active = d.eq_active[worldid, n_id]
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    equality_connect[irow].worldid = worldid
    is_site = int(m.eq_objtype[n_id] == int(types.ObjType.SITE.value))
    obj1id = m.eq_obj1id[n_id]
    obj2id = m.eq_obj2id[n_id]
    data = m.eq_data[n_id]
    for i in range(types.NREF):
      equality_connect[irow].solref[i] = m.eq_solref[n_id, i]
    for i in range(types.NIMP):
      equality_connect[irow].solimp[i] = m.eq_solimp[n_id, i]

    if is_site:
      body1id = m.site_bodyid[obj1id]
      body2id = m.site_bodyid[obj2id]
    else:
      body1id = m.eq_obj1id[n_id]
      body2id = m.eq_obj2id[n_id]

    anchor1 = wp.vec3(data[0], data[1], data[2])
    anchor2 = wp.vec3(data[3], data[4], data[5])

    pos1 = d.xmat[worldid, body1id] * anchor1 + d.xpos[worldid, body1id]
    pos2 = d.xmat[worldid, body2id] * anchor2 + d.xpos[worldid, body2id]

    if is_site:
      pos1 = d.site_xpos[worldid, obj1id]
      pos2 = d.site_xpos[worldid, obj2id]

    # error is difference in global positions
    pos = wp.vec3(0.0)
    for i in range(3):
      pos[i] = pos1[i] - pos2[i]
    equality_connect[irow].pos_aref[0] = pos[xyz_id]
    equality_connect[irow].pos_imp[0] = math.norm_l2(pos)

    # compute Jacobian difference (opposite of contact: 0 - 1)
    for i in range(m.nv):
      jacp1, _ = _jac(m, d, worldid, pos1, xyz_id, body1id, i)
      jacp2, _ = _jac(m, d, worldid, pos2, xyz_id, body2id, i)
      equality_connect[irow].J[i] = jacp1 - jacp2

    equality_connect[irow].invweight[0] = m.body_invweight0[body1id, 0] + m.body_invweight0[body2id, 0]


@wp.kernel
def _efc_equality_weld(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), eq_id: wp.array(dtype=wp.int32), equality_weld: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  c_id = id // 6
  spatial_id = id % 6
  xyz_id = id % 3
  n_id = eq_id[c_id]

  active = d.eq_active[worldid, n_id]
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    equality_weld[irow].worldid = worldid
    is_site = int(m.eq_objtype[n_id] == int(types.ObjType.SITE.value))
    obj1id = m.eq_obj1id[n_id]
    obj2id = m.eq_obj2id[n_id]
    body1id = m.eq_obj1id[n_id]
    body2id = m.eq_obj2id[n_id]
    data = m.eq_data[n_id]
    for i in range(types.NREF):
      equality_weld[irow].solref[i] = m.eq_solref[n_id, i]
    for i in range(types.NIMP):
      equality_weld[irow].solimp[i] = m.eq_solimp[n_id, i]

    if is_site:
      body1id = m.site_bodyid[obj1id]
      body2id = m.site_bodyid[obj2id]
    else:
      body1id = m.eq_obj1id[n_id]
      body2id = m.eq_obj2id[n_id]

    anchor1 = wp.vec3(data[0], data[1], data[2])
    anchor2 = wp.vec3(data[3], data[4], data[5])
    relpose = wp.quat(data[6], data[7], data[8], data[9])
    torquescale = data[10]

    # error is difference in global position and orientation
    pos1 = d.xmat[worldid, body1id] * anchor2 + d.xpos[worldid, body1id]
    pos2 = d.xmat[worldid, body2id] * anchor1 + d.xpos[worldid, body2id]

    if is_site:
      pos1 = d.site_xpos[worldid, obj1id]
      pos2 = d.site_xpos[worldid, obj2id]

    # compute orientation error: neg(q1) * q0 * relpose (axis components only)
    quat = math.mul_quat(d.xquat[worldid, body1id], relpose)
    quat1 = math.quat_inv(d.xquat[worldid, body2id])

    if is_site:
      quat = math.mul_quat(d.xquat[worldid, body1id], m.site_quat[obj1id])
      quat1 = math.quat_inv(math.mul_quat(d.xquat[worldid, body2id], m.site_quat[obj2id]))

    mulquat = math.mul_quat(quat1, quat)
    crot = wp.vec3(mulquat[1], mulquat[2], mulquat[3])  # copy axis components

    pos_imp = 0.0
    for i in range(3):
      cpos = pos1[i] - pos2[i]
      pos_imp += cpos * cpos + (crot[i] * torquescale) ** 2.0
    equality_weld[irow].pos_imp[0] = wp.sqrt(pos_imp)

    # compute Jacobian difference (opposite of contact: 0 - 1)
    for i in range(m.nv):
      if spatial_id < 3:
        jacp1, _ = _jac(m, d, worldid, pos1, xyz_id, body1id, i)
        jacp2, _ = _jac(m, d, worldid, pos2, xyz_id, body2id, i)
        jacdifp = jacp1 - jacp2
        equality_weld[irow].J[i] = jacdifp
      else:
        jacdifr = wp.vec3(0.0)
        for xyz in range(3):
          _, jacr1 = _jac(m, d, worldid, pos1, xyz, body1id, i)
          _, jacr2 = _jac(m, d, worldid, pos2, xyz, body2id, i)
          jacdifr[xyz] = (jacr1 - jacr2) * torquescale
        # correct rotation Jacobian: 0.5 * neg(q1) * (jac0-jac1) * q0 * relpose
        temp_quat = wp.quat(
          -quat1[1] * jacdifr[0] - quat1[2] * jacdifr[1] - quat1[3] * jacdifr[2],
          quat1[0] * jacdifr[0] + quat1[2] * jacdifr[2] - quat1[3] * jacdifr[1],
          quat1[0] * jacdifr[1] + quat1[3] * jacdifr[0] - quat1[1] * jacdifr[2],
          quat1[0] * jacdifr[2] + quat1[1] * jacdifr[1] - quat1[2] * jacdifr[0],
        )
        equality_weld[irow].J[i] = 0.5 * math.mul_quat(temp_quat, quat)[1 + xyz_id]

    if spatial_id < 3:
      equality_weld[irow].pos_aref[0] = pos1[xyz_id] - pos2[xyz_id]
      equality_weld[irow].invweight[0] = m.body_invweight0[body1id, 0] + m.body_invweight0[body2id, 0]
    else:
      equality_weld[irow].pos_aref[0] = crot[xyz_id] * torquescale
      equality_weld[irow].invweight[0] = m.body_invweight0[body1id, 1] + m.body_invweight0[body2id, 1]


@wp.kernel
def _efc_equality_joint(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), eq_id: wp.array(dtype=wp.int32), equality_joint: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  n_id = eq_id[id]

  active = d.eq_active[worldid, n_id]
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    equality_joint[irow].worldid = worldid
    obj1id = wp.int32(m.eq_obj1id[n_id])
    obj2id = wp.int32(m.eq_obj2id[n_id])
    dofadr1, dofadr2 = m.jnt_dofadr[obj1id], m.jnt_dofadr[obj2id]
    qposadr1, qposadr2 = m.jnt_qposadr[obj1id], m.jnt_qposadr[obj2id]
    data = m.eq_data[n_id]
    for i in range(types.NREF):
      equality_joint[irow].solref[i] = m.eq_solref[n_id, i]
    for i in range(types.NIMP):
      equality_joint[irow].solimp[i] = m.eq_solimp[n_id, i]

    pos1, pos2 = d.qpos[worldid, qposadr1], d.qpos[worldid, qposadr2]
    ref1, ref2 = m.qpos0[qposadr1], m.qpos0[qposadr2]
    dif = (pos2 - ref2) * float(obj2id > -1)
    pos = pos1 - ref1
    for i in range(5):
      pos -= data[i] * wp.pow(dif, float(i))
    equality_joint[irow].pos_aref[0] = pos
    equality_joint[irow].pos_imp[0] = pos

    deriv = float(0.0)
    for i in range(4):
      deriv += data[i + 1] * wp.pow(dif, float(i)) * float(i + 1) * float(obj2id > -1)
    equality_joint[irow].J[dofadr2] = -deriv
    equality_joint[irow].J[dofadr1] = 1.0

    invweight = m.dof_invweight0[dofadr1]
    invweight += m.dof_invweight0[dofadr2] * float(obj2id > -1)
    equality_joint[irow].invweight[0] = invweight


@wp.kernel
def _efc_dof_friction(m: types.Model, i_c: wp.array(dtype=wp.int32), dof_id: wp.array(dtype=wp.int32), dof_friction: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  n_id = dof_id[id]
  irow = wp.atomic_add(i_c, 0, 1)

  dof_friction[irow].worldid = worldid
  dof_friction[irow].frictionloss[0] = m.dof_frictionloss[n_id]
  dof_friction[irow].invweight[0] = m.dof_invweight0[n_id]
  for i in range(types.NREF):
    dof_friction[irow].solref[i] = m.dof_solref[n_id, i]
  for i in range(types.NIMP):
    dof_friction[irow].solimp[i] = m.dof_solimp[n_id, i]
  dof_friction[irow].J[n_id] = 1.0


@wp.kernel
def _efc_limit_ball(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), jnt_id: wp.array(dtype=wp.int32), limit_ball: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  n_id = jnt_id[id]

  jntquat = wp.quat(
    d.qpos[worldid, m.jnt_qposadr[n_id]],
    d.qpos[worldid, m.jnt_qposadr[n_id] + 1],
    d.qpos[worldid, m.jnt_qposadr[n_id] + 2],
    d.qpos[worldid, m.jnt_qposadr[n_id] + 3]
  )
  axis, angle = math.quat_to_axis_angle(jntquat)
  # ball rotation angle is always positive
  angle = wp.norm_l2(axis * angle)
  axis = wp.normalize(axis * angle)
  pos = wp.max(m.jnt_range[n_id, 0], m.jnt_range[n_id, 1]) - angle - m.jnt_margin[n_id]

  active = pos < 0
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    limit_ball[irow].worldid = worldid
    for i in range(types.NREF):
      limit_ball[irow].solref[i] = m.jnt_solref[n_id, i]
    for i in range(types.NIMP):
      limit_ball[irow].solimp[i] = m.jnt_solimp[n_id, i]
    limit_ball[irow].margin[0] = m.jnt_margin[n_id]
    limit_ball[irow].invweight[0] = m.dof_invweight0[m.jnt_dofadr[n_id]]
    limit_ball[irow].pos_imp[0] = pos
    for i in range(3):
      limit_ball[irow].J[m.jnt_dofadr[n_id] + i] = -axis[i]
    limit_ball[irow].pos_aref[0] = pos


@wp.kernel
def _efc_limit_slide_hinge(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), jnt_id: wp.array(dtype=wp.int32), limit_slide_hinge: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  n_id = jnt_id[id]

  qpos = d.qpos[worldid, m.jnt_qposadr[n_id]]
  dist_min, dist_max = qpos - m.jnt_range[n_id][0], m.jnt_range[n_id][1] - qpos
  pos = wp.min(dist_min, dist_max) - m.jnt_margin[n_id]
  active = pos < 0
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    limit_slide_hinge[irow].worldid = worldid
    limit_slide_hinge[irow].pos_imp[0] = pos
    for i in range(types.NREF):
      limit_slide_hinge[irow].solref[i] = m.jnt_solref[n_id, i]
    for i in range(types.NIMP):
      limit_slide_hinge[irow].solimp[i] = m.jnt_solimp[n_id, i]
    limit_slide_hinge[irow].margin[0] = m.jnt_margin[n_id]
    limit_slide_hinge[irow].invweight[0] = m.dof_invweight0[m.jnt_dofadr[n_id]]
    limit_slide_hinge[irow].J[m.jnt_dofadr[n_id]] = float(dist_min < dist_max) * 2.0 - 1.0
    limit_slide_hinge[irow].pos_aref[0] = pos


@wp.kernel
def _efc_contact_frictionless(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), con_id: wp.array(dtype=wp.int32), nrow: wp.int32, contact_frictionless: wp.array(dtype=_Efc)):

  id = wp.tid()
  n_id = con_id[id]
  worldid = contact_frictionless[nrow + id].worldid

  body1 = m.geom_bodyid[d.contact.geom[worldid, n_id, 0]]
  body2 = m.geom_bodyid[d.contact.geom[worldid, n_id, 1]]
  pos = d.contact.dist[worldid, n_id] - d.contact.includemargin[worldid, n_id]

  active = pos < 0
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    contact_frictionless[irow].worldid = worldid
    for i in range(types.NREF):
      contact_frictionless[irow].solref[i] = d.contact.solref[worldid, n_id, i]
    for i in range(types.NIMP):
      contact_frictionless[irow].solimp[i] = d.contact.solimp[worldid, n_id, i]
    contact_frictionless[irow].margin[0] = d.contact.includemargin[worldid, n_id]
    contact_frictionless[irow].invweight[0] = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]
    contact_frictionless[irow].pos_imp[0] = pos
    for i in range(m.nv):
      for xyz in range(3):
        jacp1, _ = _jac(m, d, worldid, d.contact.pos[worldid, n_id], xyz, body1, i)
        jacp2, _ = _jac(m, d, worldid, d.contact.pos[worldid, n_id], xyz, body2, i)
        contact_frictionless[irow].J[i] += d.contact.frame[worldid, n_id, xyz] * (jacp2 - jacp1)
    contact_frictionless[irow].pos_aref[0] = pos


@wp.kernel
def _efc_contact_pyramidal(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), con_id: wp.array(dtype=wp.int32), con_dim_id: wp.array(dtype=wp.int32), nrow: wp.int32, contact_pyramidal: wp.array(dtype=_Efc)):

  id = wp.tid()
  n_id = con_id[id]
  worldid = contact_pyramidal[nrow + id].worldid
  con_dim = con_dim_id[id]

  pos = d.contact.dist[worldid, n_id] - d.contact.includemargin[worldid, n_id]

  body1 = m.geom_bodyid[d.contact.geom[worldid, n_id, 0]]
  body2 = m.geom_bodyid[d.contact.geom[worldid, n_id, 1]]

  # pyramidal has common invweight across all edges
  invweight = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]
  invweight = invweight + d.contact.friction[worldid, n_id, 0] * d.contact.friction[worldid, n_id, 0] * invweight

  active = pos < 0
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    contact_pyramidal[irow].worldid = worldid
    contact_pyramidal[irow].pos_imp[0] = pos
    for i in range(types.NREF):
      contact_pyramidal[irow].solref[i] = d.contact.solref[worldid, n_id, i]
    for i in range(types.NIMP):
      contact_pyramidal[irow].solimp[i] = d.contact.solimp[worldid, n_id, i]
    contact_pyramidal[irow].margin[0] = d.contact.includemargin[worldid, n_id]
    contact_pyramidal[irow].invweight[0] = invweight * 2.0 * d.contact.friction[worldid, n_id, 0] * d.contact.friction[worldid, n_id, 0] / m.opt.impratio
    for i in range(m.nv):
      diff_0 = float(0.0)
      diff_i = float(0.0)
      for xyz in range(3):
        jac1p, jac1r = _jac(m, d, worldid, d.contact.pos[worldid, n_id], xyz, body1, i)
        jac2p, jac2r = _jac(m, d, worldid, d.contact.pos[worldid, n_id], xyz, body2, i)
        diff_0 += d.contact.frame[worldid, n_id, xyz] * (jac2p - jac1p)
        if con_dim < 3:
          diff_i += d.contact.frame[worldid, n_id, con_dim * 3 + xyz] * (jac2p - jac1p)
        else:
          diff_i += d.contact.frame[worldid, n_id, (con_dim - 3) * 3 + xyz] * (jac2r - jac1r)
      if id % 2 == 0:
        contact_pyramidal[irow].J[i] = diff_0 + diff_i * d.contact.friction[worldid, n_id, con_dim - 1]
      else:
        contact_pyramidal[irow].J[i] = diff_0 - diff_i * d.contact.friction[worldid, n_id, con_dim - 1]
    contact_pyramidal[irow].pos_aref[0] = pos


@wp.kernel
def _efc_contact_elliptic(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), con_id: wp.array(dtype=wp.int32), con_dim_id: wp.array(dtype=wp.int32), nrow: wp.int32, contact_elliptic: wp.array(dtype=_Efc)):

  id = wp.tid()
  n_id = con_id[id]
  worldid = contact_elliptic[nrow + id].worldid
  con_dim = con_dim_id[id]

  pos = d.contact.dist[worldid, n_id] - d.contact.includemargin[worldid, n_id]

  active = pos < 0
  obj1id = m.geom_bodyid[d.contact.geom[worldid, n_id, 0]]
  obj2id = m.geom_bodyid[d.contact.geom[worldid, n_id, 1]]
  invweight = m.body_invweight0[obj1id, 0] + m.body_invweight0[obj2id, 0]

  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    for i in range(types.NIMP):
      contact_elliptic[irow].solimp[i] = d.contact.solimp[worldid, n_id, i]
    contact_elliptic[irow].margin[0] = d.contact.includemargin[worldid, n_id]
    contact_elliptic[irow].pos_imp[0] = pos
    if con_dim == 0:
      contact_elliptic[irow].invweight[0] = invweight
      for i in range(types.NREF):
        contact_elliptic[irow].solref[i] = d.contact.solref[worldid, n_id, i]
    else:
      invweight_factor = invweight / m.opt.impratio
      contact_elliptic[irow].invweight[0] = invweight_factor * (
        d.contact.friction[worldid, n_id, 0] * d.contact.friction[worldid, n_id, 0] /
        (d.contact.friction[worldid, n_id, con_dim - 1] * d.contact.friction[worldid, n_id, con_dim - 1])
      )
      use_solreffriction = False
      for i in range(types.NREF):
        if (d.contact.solreffriction[worldid, n_id, i] != 0.0):
          use_solreffriction = True
      if use_solreffriction:
        for i in range(types.NREF):
          contact_elliptic[irow].solref[i] = d.contact.solreffriction[worldid, n_id, i]
      else:
        contact_elliptic[irow].solref[i] = d.contact.solref[worldid, n_id, i]
    for i in range(m.nv):
      for xyz in range(3):
        jac1p, jac1r = _jac(m, d, worldid, d.contact.pos[worldid, n_id], xyz, obj1id, i)
        jac2p, jac2r = _jac(m, d, worldid, d.contact.pos[worldid, n_id], xyz, obj2id, i)
        if con_dim < 3:
          contact_elliptic[irow].J[i] += d.contact.frame[worldid, n_id, con_dim * 3 + xyz] * (jac2p - jac1p)
        else:
          contact_elliptic[irow].J[i] += d.contact.frame[worldid, n_id, (con_dim - 3) * 3 + xyz] * (jac2r - jac1r)

    if con_dim == 0:
      contact_elliptic[irow].pos_aref[0] = pos


def make_constraint(m: types.Model, d: types.Data):
  """Creates constraint jacobians and other supporting data."""

  nrow = 0
  i_c = wp.zeros(1, dtype=int)
  if not( m.opt.disableflags & types.DisableBit.CONSTRAINT):
    eq_type = m.eq_type.numpy()
    dof_hasfrictionloss = m.dof_hasfrictionloss.numpy()
    jnt_type = m.jnt_type.numpy()
    jnt_limited = m.jnt_limited.numpy()
    con_dim = d.contact.dim.numpy()

    # TODO improve this, preallocate and precompute as much as possible
    temp_eq_connect_id = ()
    temp_eq_weld_id = ()
    temp_eq_joint_id = ()
    temp_dof_friction_id = ()
    temp_jmt_ball_id = ()
    temp_jmt_slide_hinge_id = ()
    temp_con_frictionless_id = ()
    temp_con_id = ()
    temp_con_dim_id = ()
    worldid = ()
    for i in range(m.neq):
      if (eq_type[i] == types.EqType.CONNECT):
        temp_eq_connect_id += (i,)
      if (eq_type[i] == types.EqType.WELD):
        temp_eq_weld_id += (i,)
      if (eq_type[i] == types.EqType.JOINT):
        temp_eq_joint_id += (i,)
    for i in range(m.nv):
      if (dof_hasfrictionloss[i]):
        temp_dof_friction_id += (i,)
    for i in range(m.njnt):
      if ((jnt_type[i] == types.JointType.BALL) and jnt_limited[i]):
        temp_jmt_ball_id += (i,)
      if ((jnt_type[i] == types.JointType.SLIDE or jnt_type[i] == types.JointType.HINGE) and jnt_limited[i]):
        temp_jmt_slide_hinge_id += (i,)
    worldid += tuple(w for w in range(d.nworld) for _ in range(3 * len(temp_eq_connect_id)))
    worldid += tuple(w for w in range(d.nworld) for _ in range(6 * len(temp_eq_weld_id)))
    worldid += tuple(w for w in range(d.nworld) for _ in range(len(temp_eq_joint_id)))
    worldid += tuple(w for w in range(d.nworld) for _ in range(len(temp_dof_friction_id)))
    worldid += tuple(w for w in range(d.nworld) for _ in range(len(temp_jmt_ball_id)))
    worldid += tuple(w for w in range(d.nworld) for _ in range(len(temp_jmt_slide_hinge_id)))
    for i in range(d.ncon):
      for w in range(d.nworld):
        if (con_dim[w, i] == 1):
          temp_con_frictionless_id += (i,)
          worldid += (w,)
    for condim in (3, 4, 6):
      for i in range(d.ncon):
        for w in range(d.nworld):
          if (con_dim[w, i] == condim):
            if m.opt.cone == types.ConeType.ELLIPTIC:
              for j in range(condim):
                temp_con_id += (i,)
                temp_con_dim_id += (j,)
                worldid += (w,)
            else:
              for j in range(condim-1):
                temp_con_id += (i,i,)
                temp_con_dim_id += (j+1,j+1,)
                worldid += (w,w,)

    eq_connect_id = wp.array(temp_eq_connect_id, dtype=wp.int32)
    eq_weld_id = wp.array(temp_eq_weld_id, dtype=wp.int32)
    eq_joint_id = wp.array(temp_eq_joint_id, dtype=wp.int32)
    dof_friction_id = wp.array(temp_dof_friction_id, dtype=wp.int32)
    jnt_ball_id = wp.array(temp_jmt_ball_id, dtype=wp.int32)
    jnt_slide_hinge_id = wp.array(temp_jmt_slide_hinge_id, dtype=wp.int32)
    con_frictionless_id = wp.array(temp_con_frictionless_id, dtype=wp.int32)
    con_id = wp.array(temp_con_id, dtype=wp.int32)
    com_dim_id = wp.array(temp_con_dim_id, dtype=wp.int32)

    # Allocate the constraint rows
    ntotalrow = (
      3 * eq_connect_id.size * d.nworld + 6 * eq_weld_id.size * d.nworld + eq_joint_id.size * d.nworld +
      dof_friction_id.size * d.nworld + jnt_ball_id.size * d.nworld + jnt_slide_hinge_id.size * d.nworld +
      con_frictionless_id.size + con_id.size
    )
    temp_efcs = []
    for i in range(ntotalrow):
      row = _Efc()
      row.worldid = worldid[i]
      row.J = wp.zeros((m.nv), dtype=wp.float32)
      row.pos_aref = wp.zeros(shape=(1), dtype=wp.float32)
      row.pos_imp = wp.zeros(shape=(1), dtype=wp.float32)
      row.invweight = wp.zeros(shape=(1), dtype=wp.float32)
      row.solref = wp.zeros(shape=(types.NREF), dtype=wp.float32)
      row.solimp = wp.zeros(shape=(types.NIMP), dtype=wp.float32)
      row.margin = wp.zeros(shape=(1), dtype=wp.float32)
      row.frictionloss = wp.zeros(shape=(1), dtype=wp.float32)
      temp_efcs.append(row)
    efcs = wp.array(temp_efcs, ndim=2, dtype=_Efc)

    if not (m.opt.disableflags & types.DisableBit.EQUALITY) and eq_connect_id.size != 0:
      wp.launch(_efc_equality_connect, dim=(d.nworld, 3 * eq_connect_id.size), inputs=[m, d, i_c, eq_connect_id], outputs=[efcs])
      nrow += 3 * eq_connect_id.size * d.nworld

    if not (m.opt.disableflags & types.DisableBit.EQUALITY) and eq_weld_id.size != 0:
      wp.launch(_efc_equality_weld, dim=(d.nworld, 6 * eq_weld_id.size), inputs=[m, d, i_c, eq_weld_id], outputs=[efcs])
      nrow += 6 * eq_weld_id.size * d.nworld

    if not (m.opt.disableflags & types.DisableBit.EQUALITY) and eq_joint_id.size != 0:
      wp.launch(_efc_equality_joint, dim=(d.nworld, eq_joint_id.size), inputs=[m, d, i_c, eq_joint_id], outputs=[efcs])
      nrow += eq_joint_id.size * d.nworld

    if not (m.opt.disableflags & types.DisableBit.FRICTIONLOSS) and (dof_friction_id.size != 0):
      wp.launch(_efc_dof_friction, dim=(d.nworld, dof_friction_id.size), inputs=[m, i_c, dof_friction_id, efcs])
      nrow += dof_friction_id.size * d.nworld

    if not (m.opt.disableflags & types.DisableBit.LIMIT) and (jnt_ball_id.size != 0):
      wp.launch(_efc_limit_ball, dim=(d.nworld, jnt_ball_id.size), inputs=[m, d, i_c, jnt_ball_id, efcs])
      nrow += jnt_ball_id.size * d.nworld

    if not (m.opt.disableflags & types.DisableBit.LIMIT) and (jnt_slide_hinge_id.size != 0):
      wp.launch(_efc_limit_slide_hinge, dim=(d.nworld, jnt_slide_hinge_id.size), inputs=[m, d, i_c, jnt_slide_hinge_id, efcs])
      nrow += jnt_slide_hinge_id.size * d.nworld

    if (con_frictionless_id.size != 0):
      wp.launch(_efc_contact_frictionless, dim=con_frictionless_id.size, inputs=[m, d, i_c, con_frictionless_id, nrow, efcs])
      nrow += con_frictionless_id.size

    if (con_id.size != 0):
      if m.opt.cone == types.ConeType.ELLIPTIC:
        wp.launch(_efc_contact_elliptic, dim=con_id.size, inputs=[m, d, i_c, con_id, com_dim_id, nrow, efcs])
      else:
        wp.launch(_efc_contact_pyramidal, dim=con_id.size, inputs=[m, d, i_c, con_id, com_dim_id, nrow, efcs])
      nrow += con_id.size

  i_c_np = int(i_c.numpy()[0])
  if i_c_np == 0:
    d.efc_J = wp.empty((0, m.nv))
    d.efc_D = wp.empty(0)
    d.efc_aref = wp.empty(0)
    d.efc_frictionloss = wp.empty(0)
    d.efc_pos = wp.empty(0)
    d.efc_margin = wp.empty(0)
    return d

  refsafe = (not m.opt.disableflags & types.DisableBit.REFSAFE)
  wp.launch(_update_contact_data, dim=i_c_np, inputs=[m, d, efcs, refsafe])

  return d

