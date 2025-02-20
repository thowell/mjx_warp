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
    #TODO: temporary change to test without the jacobian computation
    aref += -b * (efc.J[i] * d.qvel[worldid, i])
    #d.efc_J[worldid, id, i] = efc.J[i]
  aref -= k * imp * efc.pos_aref[0]
  d.efc_D[worldid, id] = 1.0 / r
  d.efc_aref[worldid, id] = aref
  d.efc_pos[worldid, id]  = efc.pos_aref[0] + efc.margin[0]
  d.efc_margin[worldid, id]  = efc.margin[0]
  d.efc_frictionloss[worldid, id]  = efc.frictionloss[0]


@wp.func
def _jac(m: types.Model, d: types.Data, point: wp.vec3, bodyid: wp.int32):
  #fn = lambda carry, b: b if carry is None else b + carry
  #mask = (jp.arange(m.nbody) == bodyid) * 1
  #mask = scan.body_tree(m, fn, 'b', 'b', mask, reverse=True)
  #mask = mask[jp.array(m.dof_bodyid)] > 0

  #offset = point - d.subtree_com[jp.array(m.body_rootid)[body_id]]
  #jacp = jax.vmap(lambda a, b=offset: a[3:] + jp.cross(a[:3], b))(d.cdof)
  #jacp = jax.vmap(jp.multiply)(jacp, mask)
  #jacr = jax.vmap(jp.multiply)(d.cdof[:, :3], mask)

  return wp.vec3(0.0), wp.vec3(0.0)


@wp.kernel
def _efc_equality_connect(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), eq_id: wp.array(dtype=wp.int32), equality_connect: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  c_id = id // 3
  xyz_id = id % 3
  n_id = eq_id[c_id]

  active = d.eq_active[worldid, n_id]
  if active:
    id = wp.atomic_add(i_c, 0, 1)
    equality_connect[id].worldid = worldid
    is_site = int(m.eq_objtype[n_id] == int(types.ObjType.SITE.value))
    obj1id = m.eq_obj1id[n_id]
    obj2id = m.eq_obj2id[n_id]
    data = m.eq_data[n_id]
    for i in range(types.NREF):
      equality_connect[id].solref[i] = m.eq_solref[n_id, i]
    for i in range(types.NIMP):
      equality_connect[id].solimp[i] = m.eq_solimp[n_id, i]

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
    equality_connect[id].pos_aref[0] = pos[xyz_id]
    equality_connect[id].pos_imp[0] = math.norm_l2(pos)

    # compute Jacobian difference (opposite of contact: 0 - 1)
    jacp1, _ = _jac(m, d, pos1, body1id)
    jacp2, _ = _jac(m, d, pos2, body2id)
    # TODO: Update jacobian
    #for i in range(m.nv):
      #equality_connect[id].J[i] = jacp1[i] - jacp2[i]

    equality_connect[id].invweight[0] = m.body_invweight0[body1id, 0] + m.body_invweight0[body2id, 0]


@wp.kernel
def _efc_equality_weld(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), eq_id: wp.array(dtype=wp.int32), equality_weld: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  c_id = id // 6
  xyz_id = id % 6
  n_id = eq_id[c_id]

  active = d.eq_active[worldid, n_id]
  if active:
    id = wp.atomic_add(i_c, 0, 1)
    equality_weld[id].worldid = worldid
    is_site = int(m.eq_objtype[n_id] == int(types.ObjType.SITE.value))
    obj1id = m.eq_obj1id[n_id]
    obj2id = m.eq_obj2id[n_id]
    body1id = m.eq_obj1id[n_id]
    body2id = m.eq_obj2id[n_id]
    data = m.eq_data[n_id]
    for i in range(types.NREF):
      equality_weld[id].solref[i] = m.eq_solref[n_id, i]
    for i in range(types.NIMP):
      equality_weld[id].solimp[i] = m.eq_solimp[n_id, i]

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

    # compute Jacobian difference (opposite of contact: 0 - 1)
    jacp1, jacr1 = _jac(m, d, pos1, body1id)
    jacp2, jacr2 = _jac(m, d, pos2, body2id)
    jacdifp = jacp1 - jacp2
    jacdifr = (jacr1 - jacr2) * torquescale

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
    equality_weld[id].pos_imp[0] = wp.sqrt(pos_imp)

    # TODO update jacobian
    # correct rotation Jacobian: 0.5 * neg(q1) * (jac0-jac1) * q0 * relpose
    #jacdifr = 0.5 * math.mul_quat(math.quat_mul_axis(quat1, jacdifr), quat)[1:]
    #j = jp.concatenate((jacdifp.T, jacdifr.T))

    if xyz_id < 3:
      equality_weld[id].pos_aref[0] = pos1[xyz_id] - pos2[xyz_id]
      equality_weld[id].invweight[0] = m.body_invweight0[body1id, 0] + m.body_invweight0[body2id, 0]
    else:
      equality_weld[id].pos_aref[0] = crot[xyz_id- 3 ] * torquescale
      equality_weld[id].invweight[0] = m.body_invweight0[body1id, 1] + m.body_invweight0[body2id, 1]


@wp.kernel
def _efc_equality_joint(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), eq_id: wp.array(dtype=wp.int32), equality_joint: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  n_id = eq_id[id]

  active = d.eq_active[worldid, n_id]
  if active:
    id = wp.atomic_add(i_c, 0, 1)
    equality_joint[id].worldid = worldid
    obj1id = wp.int32(m.eq_obj1id[n_id])
    obj2id = wp.int32(m.eq_obj2id[n_id])
    dofadr1, dofadr2 = m.jnt_dofadr[obj1id], m.jnt_dofadr[obj2id]
    qposadr1, qposadr2 = m.jnt_qposadr[obj1id], m.jnt_qposadr[obj2id]
    data = m.eq_data[n_id]
    for i in range(types.NREF):
      equality_joint[id].solref[i] = m.eq_solref[n_id, i]
    for i in range(types.NIMP):
      equality_joint[id].solimp[i] = m.eq_solimp[n_id, i]

    pos1, pos2 = d.qpos[worldid, qposadr1], d.qpos[worldid, qposadr2]
    ref1, ref2 = m.qpos0[qposadr1], m.qpos0[qposadr2]
    dif = (pos2 - ref2) * float(obj2id > -1)
    pos = pos1 - ref1
    for i in range(5):
      pos -= data[i] * wp.pow(dif, float(i))
    equality_joint[id].pos_aref[0] = pos
    equality_joint[id].pos_imp[0] = pos

    # TODO update jacobian
    #deriv = wp.dot(data[1:5], dif_power[:4] * wp.array([1, 2, 3, 4])) * (obj2id > -1)
    # = wp.zeros((m.nv)).at[dofadr2].set(-deriv).at[dofadr1].set(1.0)

    invweight = m.dof_invweight0[dofadr1]
    invweight += m.dof_invweight0[dofadr2] * float(obj2id > -1)
    equality_joint[id].invweight[0] = invweight


@wp.kernel
def _efc_dof_friction(m: types.Model, i_c: wp.array(dtype=wp.int32), dof_id: wp.array(dtype=wp.int32), dof_friction: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  n_id = dof_id[id]
  id = wp.atomic_add(i_c, 0, 1)

  dof_friction[id].worldid = worldid
  dof_friction[id].frictionloss[0] = m.dof_frictionloss[n_id]
  dof_friction[id].invweight[0] = m.dof_invweight0[n_id]
  for i in range(types.NREF):
    dof_friction[id].solref[i] = m.dof_solref[n_id, i]
  for i in range(types.NIMP):
    dof_friction[id].solimp[i] = m.dof_solimp[n_id, i]
  dof_friction[id].J[n_id] = 1.0


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
    id = wp.atomic_add(i_c, 0, 1)
    limit_ball[id].worldid = worldid
    for i in range(types.NREF):
      limit_ball[id].solref[i] = m.jnt_solref[n_id, i]
    for i in range(types.NIMP):
      limit_ball[id].solimp[i] = m.jnt_solimp[n_id, i]
    limit_ball[id].margin[0] = m.jnt_margin[n_id]
    limit_ball[id].invweight[0] = m.dof_invweight0[m.jnt_dofadr[n_id]]
    limit_ball[id].pos_imp[0] = pos
    # TODO update jacobian
    #j = wp.zeros(m.nv).at[jp.arange(3) + dofadr].set(-axis)
    limit_ball[id].pos_aref[0] = pos


@wp.kernel
def _efc_limit_slide_hinge(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), jnt_id: wp.array(dtype=wp.int32), limit_slide_hinge: wp.array(dtype=_Efc)):

  worldid, id = wp.tid()
  n_id = jnt_id[id]

  qpos = d.qpos[worldid, m.jnt_qposadr[n_id]]
  dist_min, dist_max = qpos - m.jnt_range[n_id][0], m.jnt_range[n_id][1] - qpos
  pos = wp.min(dist_min, dist_max) - m.jnt_margin[n_id]
  active = pos < 0
  if active:
    id = wp.atomic_add(i_c, 0, 1)
    limit_slide_hinge[id].worldid = worldid
    limit_slide_hinge[id].pos_imp[0] = pos
    for i in range(types.NREF):
      limit_slide_hinge[id].solref[i] = m.jnt_solref[n_id, i]
    for i in range(types.NIMP):
      limit_slide_hinge[id].solimp[i] = m.jnt_solimp[n_id, i]
    limit_slide_hinge[id].margin[0] = m.jnt_margin[n_id]
    limit_slide_hinge[id].invweight[0] = m.dof_invweight0[m.jnt_dofadr[n_id]]
    # TODO update jacobian
    #j = wp.zeros(m.nv).at[dofadr].set((dist_min < dist_max) * 2 - 1)
    limit_slide_hinge[id].pos_aref[0] = pos


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
    id = wp.atomic_add(i_c, 0, 1)
    contact_frictionless[id].worldid = worldid
    for i in range(types.NREF):
      contact_frictionless[id].solref[i] = d.contact.solref[worldid, n_id, i]
    for i in range(types.NIMP):
      contact_frictionless[id].solimp[i] = d.contact.solimp[worldid, n_id, i]
    contact_frictionless[id].margin[0] = d.contact.includemargin[worldid, n_id]
    contact_frictionless[id].invweight[0] = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]
    contact_frictionless[id].pos_imp[0] = pos
    # TODO update jacobian
    #jac1p, _ = _jac(m, d, d.contact.pos[worldid, n_id], body1)
    #jac2p, _ = _jac(m, d, d.contact.pos[worldid, n_id], body2)
    #j = (d.contact.frame[worldid, n_id] @ (jac2p - jac1p).T)[0]
    contact_frictionless[id].pos_aref[0] = pos


@wp.kernel
def _efc_contact_pyramidal(m: types.Model, d: types.Data, i_c: wp.array(dtype=wp.int32), con_id: wp.array(dtype=wp.int32), nrow: wp.int32, contact_pyramidal: wp.array(dtype=_Efc)):

  id = wp.tid()
  n_id = con_id[id]
  worldid = contact_pyramidal[nrow + id].worldid

  pos = d.contact.dist[worldid, n_id] - d.contact.includemargin[worldid, n_id]

  body1 = m.geom_bodyid[d.contact.geom[worldid, n_id, 0]]
  body2 = m.geom_bodyid[d.contact.geom[worldid, n_id, 1]]
  jac1p, jac1r = _jac(m, d, d.contact.pos[worldid, n_id], body1)
  jac2p, jac2r = _jac(m, d, d.contact.pos[worldid, n_id], body2)

  # pyramidal has common invweight across all edges
  invweight = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]
  invweight = invweight + d.contact.friction[worldid, n_id, 0] * d.contact.friction[worldid, n_id, 0] * invweight

  active = pos < 0
  if active:
    id = wp.atomic_add(i_c, 0, 1)
    contact_pyramidal[id].worldid = worldid
    contact_pyramidal[id].pos_imp[0] = pos
    for i in range(types.NREF):
      contact_pyramidal[id].solref[i] = d.contact.solref[worldid, n_id, i]
    for i in range(types.NIMP):
      contact_pyramidal[id].solimp[i] = d.contact.solimp[worldid, n_id, i]
    contact_pyramidal[id].margin[0] = d.contact.includemargin[worldid, n_id]
    contact_pyramidal[id].invweight[0] = invweight * 2.0 * d.contact.friction[worldid, n_id, 0] * d.contact.friction[worldid, n_id, 0] / m.opt.impratio
    # TODO update jacobian
    # a pair of opposing pyramid edges per friction dimension
    # repeat friction directions with positive and negative sign
    #fri = jp.repeat(c.friction[: condim - 1], 2, axis=0).at[1::2].mul(-1)
    #diff = c.frame @ (jac2p - jac1p).T
    #if condim > 3:
    #  diff = jp.concatenate((diff, (c.frame @ (jac2r - jac1r).T)), axis=0)
    # repeat condims of jacdiff to match +/- friction directions
    #j = diff[0] + jp.repeat(diff[1:condim], 2, axis=0) * fri[:, None]
    contact_pyramidal[id].pos_aref[0] = pos


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
  jac1p, jac1r = _jac(m, d, d.contact.pos[worldid, n_id], obj1id)
  jac2p, jac2r = _jac(m, d, d.contact.pos[worldid, n_id], obj2id)
  invweight = m.body_invweight0[obj1id, 0] + m.body_invweight0[obj2id, 0]

  if active:
    id = wp.atomic_add(i_c, 0, 1)
    for i in range(types.NIMP):
      contact_elliptic[id].solimp[i] = d.contact.solimp[worldid, n_id, i]
    contact_elliptic[id].margin[0] = d.contact.includemargin[worldid, n_id]
    contact_elliptic[id].pos_imp[0] = pos
    if con_dim == 0:
      contact_elliptic[id].invweight[0] = invweight
      for i in range(types.NREF):
        contact_elliptic[id].solref[i] = d.contact.solref[worldid, n_id, i]
    else:
      invweight_factor = invweight / m.opt.impratio
      contact_elliptic[id].invweight[0] = invweight_factor * (
        d.contact.friction[worldid, n_id, 0] * d.contact.friction[worldid, n_id, 0] /
        (d.contact.friction[worldid, n_id, con_dim - 1] * d.contact.friction[worldid, n_id, con_dim - 1])
      )
      use_solreffriction = False
      for i in range(types.NREF):
        if (d.contact.solreffriction[worldid, n_id, i] != 0.0):
          use_solreffriction = True
      if use_solreffriction:
        for i in range(types.NREF):
          contact_elliptic[id].solref[i] = d.contact.solreffriction[worldid, n_id, i]
      else:
        contact_elliptic[id].solref[i] = d.contact.solref[worldid, n_id, i]
    # TODO update jacobian
    #j = d.contact.frame @ (jac2p - jac1p).T
    #if d.contact.dim[worldid, n_id] > 3:
    #  j = jp.concatenate((j, (d.contact.frame @ (jac2r - jac1r).T)[: d.contact.dim[worldid, n_id] - 3]))
    if con_dim == 0:
      contact_elliptic[id].pos_aref[0] = pos


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
        wp.launch(_efc_contact_pyramidal, dim=con_id.size, inputs=[m, d, i_c, con_id, nrow, efcs])
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

