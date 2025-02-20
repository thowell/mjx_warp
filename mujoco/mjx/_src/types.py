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
from enum import Enum, IntFlag


class vec10f(wp.types.vector(length=10, dtype=wp.float32)):
  pass


vec10 = vec10f
MINIMP = 0.0001  # minimum constraint impedance
MAXIMP = 0.9999  # maximum constraint impedance
MINVAL = 1E-15
NREF = 2
NIMP = 5

class DisableBit(IntFlag):
  CONSTRAINT = 1
  EQUALITY = 2
  FRICTIONLOSS = 4
  LIMIT = 8
  CONTACT = 16
  PASSIVE = 32
  GRAVITY = 64
  CLAMPCTRL = 128
  WARMSTART = 256
  FILTERPARENT = 512
  ACTUATION = 1024
  REFSAFE = 2048
  SENSOR = 4096
  # unsupported: MIDPHASE
  EULERDAMP = 16384


class JointType(IntFlag):
  FREE = 0
  BALL = 1
  SLIDE = 2
  HINGE = 3


class ConeType(IntFlag):
  PYRAMIDAL = 0
  ELLIPTIC = 1


class EqType(IntFlag):
  CONNECT = 0
  WELD = 1
  JOINT = 2
  TENDON = 3
  # unsupported: DISTANCE


class ConstraintType(IntFlag):
  EQUALITY = 0
  FRICTION_DOF = 1
  FRICTION_TENDON = 2
  LIMIT_JOINT = 3
  LIMIT_TENDON = 4
  CONTACT_FRICTIONLESS = 5
  CONTACT_PYRAMIDAL = 6
  CONTACT_ELLIPTIC = 7

@wp.struct
class ObjType(Enum):
  UNKNOWN = 0
  BODY = 1
  XBODY = 2
  GEOM = 5
  SITE = 6
  CAMERA = 7

array2df = wp.array2d(dtype=wp.float32, ndim=2)


@wp.struct
class Option:
  gravity: wp.vec3
  is_sparse: bool # warp only
  cone: wp.int32
  disableflags: wp.int32
  impratio: wp.float32
  timestep: wp.float32


@wp.struct
class Model:
  nq: int
  nv: int
  nu: int
  nbody: int
  njnt: int
  ngeom: int
  nsite: int
  neq: int
  nmocap: int
  nM: int
  opt: Option
  qpos0: wp.array(dtype=wp.float32, ndim=1)
  qpos_spring: wp.array(dtype=wp.float32, ndim=1)
  body_tree: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_treeadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_update_tree: wp.array(dtype=wp.vec3i, ndim=1)  # warp only
  qLD_update_treeadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_tile: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_tileadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_tilesize: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_dofadr: wp.array(dtype=wp.int32, ndim=1)
  body_dofnum: wp.array(dtype=wp.int32, ndim=1)
  body_jntadr: wp.array(dtype=wp.int32, ndim=1)
  body_jntnum: wp.array(dtype=wp.int32, ndim=1)
  body_parentid: wp.array(dtype=wp.int32, ndim=1)
  body_mocapid: wp.array(dtype=wp.int32, ndim=1)
  body_pos: wp.array(dtype=wp.vec3, ndim=1)
  body_quat: wp.array(dtype=wp.quat, ndim=1)
  body_ipos: wp.array(dtype=wp.vec3, ndim=1)
  body_iquat: wp.array(dtype=wp.quat, ndim=1)
  body_rootid: wp.array(dtype=wp.int32, ndim=1)
  body_inertia: wp.array(dtype=wp.vec3, ndim=1)
  body_mass: wp.array(dtype=wp.float32, ndim=1)
  body_invweight0: wp.array(dtype=wp.float32, ndim=2)
  jnt_bodyid: wp.array(dtype=wp.int32, ndim=1)
  jnt_limited: wp.array(dtype=wp.int32, ndim=1)
  jnt_solref: wp.array(dtype=wp.float32, ndim=2)
  jnt_solimp: wp.array(dtype=wp.float32, ndim=2)
  jnt_type: wp.array(dtype=wp.int32, ndim=1)
  jnt_qposadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_dofadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_axis: wp.array(dtype=wp.vec3, ndim=1)
  jnt_pos: wp.array(dtype=wp.vec3, ndim=1)
  jnt_range: wp.array(dtype=wp.float32, ndim=2)
  jnt_margin: wp.array(dtype=wp.float32, ndim=1)
  jnt_stiffness: wp.array(dtype=wp.float32, ndim=1)
  geom_bodyid: wp.array(dtype=wp.int32, ndim=1)
  geom_pos: wp.array(dtype=wp.vec3, ndim=1)
  geom_quat: wp.array(dtype=wp.quat, ndim=1)
  site_bodyid: wp.array(dtype=wp.int32, ndim=1)
  site_pos: wp.array(dtype=wp.vec3, ndim=1)
  site_quat: wp.array(dtype=wp.quat, ndim=1)
  dof_bodyid: wp.array(dtype=wp.int32, ndim=1)
  dof_jntid: wp.array(dtype=wp.int32, ndim=1)
  dof_parentid: wp.array(dtype=wp.int32, ndim=1)
  dof_Madr: wp.array(dtype=wp.int32, ndim=1)
  dof_solref: wp.array(dtype=wp.float32, ndim=2)
  dof_solimp: wp.array(dtype=wp.float32, ndim=2)
  dof_frictionloss: wp.array(dtype=wp.float32, ndim=1)
  dof_hasfrictionloss: wp.array(dtype=wp.int32, ndim=1)
  dof_armature: wp.array(dtype=wp.float32, ndim=1)
  dof_invweight0: wp.array(dtype=wp.float32, ndim=1)
  dof_damping: wp.array(dtype=wp.float32, ndim=1)
  eq_type: wp.array(dtype=wp.int32, ndim=1)
  eq_obj1id: wp.array(dtype=wp.int32, ndim=1)
  eq_obj2id: wp.array(dtype=wp.int32, ndim=1)
  eq_objtype: wp.array(dtype=wp.int32, ndim=1)
  eq_solref: wp.array(dtype=wp.float32, ndim=2)
  eq_solimp: wp.array(dtype=wp.float32, ndim=2)
  eq_data: wp.array(dtype=wp.float32, ndim=2)
  opt: Option


@wp.struct
class Contact:
  dist: wp.array(dtype=wp.float32, ndim=2)
  pos: wp.array(dtype=wp.vec3f, ndim=2)
  frame: wp.array(dtype=wp.float32, ndim=3)
  includemargin: wp.array(dtype=wp.float32, ndim=2)
  friction: wp.array(dtype=wp.float32, ndim=3)
  solref: wp.array(dtype=wp.float32, ndim=3)
  solreffriction: wp.array(dtype=wp.float32, ndim=3)
  solimp: wp.array(dtype=wp.float32, ndim=3)
  dim: wp.array(dtype=wp.int32, ndim=2)
  geom1: wp.array(dtype=wp.int32, ndim=2)
  geom2: wp.array(dtype=wp.int32, ndim=2)
  geom: wp.array(dtype=wp.int32, ndim=3)
  efc_address: wp.array(dtype=wp.int32, ndim=2)


@wp.struct
class Data:
  nworld: int
  ncon: int
  ne: int
  nf: int
  nl: int
  nefc: int
  qpos: wp.array(dtype=wp.float32, ndim=2)
  eq_active: wp.array(dtype=wp.int32, ndim=2)
  qvel: wp.array(dtype=wp.float32, ndim=2)
  qfrc_applied: wp.array(dtype=wp.float32, ndim=2)
  mocap_pos: wp.array(dtype=wp.vec3, ndim=2)
  mocap_quat: wp.array(dtype=wp.quat, ndim=2)
  qacc: wp.array(dtype=wp.float32, ndim=2)
  xanchor: wp.array(dtype=wp.vec3, ndim=2)
  xaxis: wp.array(dtype=wp.vec3, ndim=2)
  xmat: wp.array(dtype=wp.mat33, ndim=2)
  xpos: wp.array(dtype=wp.vec3, ndim=2)
  xquat: wp.array(dtype=wp.quat, ndim=2)
  xipos: wp.array(dtype=wp.vec3, ndim=2)
  ximat: wp.array(dtype=wp.mat33, ndim=2)
  subtree_com: wp.array(dtype=wp.vec3, ndim=2)
  geom_xpos: wp.array(dtype=wp.vec3, ndim=2)
  geom_xmat: wp.array(dtype=wp.mat33, ndim=2)
  site_xpos: wp.array(dtype=wp.vec3, ndim=2)
  site_xmat: wp.array(dtype=wp.mat33, ndim=2)
  cinert: wp.array(dtype=vec10, ndim=2)
  cdof: wp.array(dtype=wp.spatial_vector, ndim=2)
  actuator_moment: wp.array(dtype=wp.float32, ndim=3)
  crb: wp.array(dtype=vec10, ndim=2)
  qM: wp.array(dtype=wp.float32, ndim=3)
  qLD: wp.array(dtype=wp.float32, ndim=3)
  qLDiagInv: wp.array(dtype=wp.float32, ndim=2)
  actuator_velocity: wp.array(dtype=wp.float32, ndim=2)
  cvel: wp.array(dtype=wp.spatial_vector, ndim=2)
  cdof_dot: wp.array(dtype=wp.spatial_vector, ndim=2)
  qfrc_bias: wp.array(dtype=wp.float32, ndim=2)
  qfrc_passive: wp.array(dtype=wp.float32, ndim=2)
  qfrc_spring: wp.array(dtype=wp.float32, ndim=2)
  qfrc_damper: wp.array(dtype=wp.float32, ndim=2)
  qfrc_actuator: wp.array(dtype=wp.float32, ndim=2)
  qfrc_smooth: wp.array(dtype=wp.float32, ndim=2)
  qacc_smooth: wp.array(dtype=wp.float32, ndim=2)
  contact: Contact
  efc_J: wp.array(dtype=wp.float32, ndim=3)
  efc_pos: wp.array(dtype=wp.float32, ndim=2)
  efc_margin: wp.array(dtype=wp.float32, ndim=2)
  efc_frictionloss: wp.array(dtype=wp.float32, ndim=2)
  efc_D: wp.array(dtype=wp.float32, ndim=2)
  efc_aref: wp.array(dtype=wp.float32, ndim=2)
