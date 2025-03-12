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
import enum
import mujoco

MJ_MINVAL = mujoco.mjMINVAL
MJ_MINIMP = mujoco.mjMINIMP  # minimum constraint impedance
MJ_MAXIMP = mujoco.mjMAXIMP  # maximum constraint impedance
MJ_NREF = mujoco.mjNREF
MJ_NIMP = mujoco.mjNIMP


class DisableBit(enum.IntFlag):
  """Disable default feature bitflags.

  Members:
    CONSTRAINT:   entire constraint solver
    EQUALITY:     equality constraints
    FRICTIONLOSS: joint and tendon frictionloss constraints
    LIMIT:        joint and tendon limit constraints
    CONTACT:      contact constraints
    PASSIVE:      passive forces
    GRAVITY:      gravitational forces
    CLAMPCTRL:    clamp control to specified range
    WARMSTART:    warmstart constraint solver
    ACTUATION:    apply actuation forces
    REFSAFE:      integrator safety: make ref[0]>=2*timestep
    SENSOR:       sensors
    EULERDAMP:    implicit damping for Euler integration
    FILTERPARENT: disable collisions between parent and child bodies
  """

  CONSTRAINT = mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
  EQUALITY = mujoco.mjtDisableBit.mjDSBL_EQUALITY
  FRICTIONLOSS = mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
  LIMIT = mujoco.mjtDisableBit.mjDSBL_LIMIT
  CONTACT = mujoco.mjtDisableBit.mjDSBL_CONTACT
  PASSIVE = mujoco.mjtDisableBit.mjDSBL_PASSIVE
  GRAVITY = mujoco.mjtDisableBit.mjDSBL_GRAVITY
  CLAMPCTRL = mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
  WARMSTART = mujoco.mjtDisableBit.mjDSBL_WARMSTART
  ACTUATION = mujoco.mjtDisableBit.mjDSBL_ACTUATION
  REFSAFE = mujoco.mjtDisableBit.mjDSBL_REFSAFE
  SENSOR = mujoco.mjtDisableBit.mjDSBL_SENSOR
  EULERDAMP = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
  FILTERPARENT = mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
  # unsupported: MIDPHASE


class TrnType(enum.IntEnum):
  """Type of actuator transmission.

  Members:
    JOINT: force on joint
    JOINTINPARENT: force on joint, expressed in parent frame
    TENDON: force on tendon (unsupported)
    SITE: force on site (unsupported)
  """

  JOINT = mujoco.mjtTrn.mjTRN_JOINT
  JOINTINPARENT = mujoco.mjtTrn.mjTRN_JOINTINPARENT
  # unsupported: SITE, TENDON, SLIDERCRANK, BODY


class DynType(enum.IntEnum):
  """Type of actuator dynamics.

  Members:
    NONE: no internal dynamics; ctrl specifies force
    INTEGRATOR: integrator: da/dt = u
    FILTER: linear filter: da/dt = (u-a) / tau
    FILTEREXACT: linear filter: da/dt = (u-a) / tau, with exact integration
    MUSCLE: piece-wise linear filter with two time constants
  """

  NONE = mujoco.mjtDyn.mjDYN_NONE
  INTEGRATOR = mujoco.mjtDyn.mjDYN_INTEGRATOR
  FILTER = mujoco.mjtDyn.mjDYN_FILTER
  FILTEREXACT = mujoco.mjtDyn.mjDYN_FILTEREXACT
  MUSCLE = mujoco.mjtDyn.mjDYN_MUSCLE
  # unsupported: USER


class GainType(enum.IntEnum):
  """Type of actuator gain.

  Members:
    FIXED: fixed gain
    AFFINE: const + kp*length + kv*velocity
    MUSCLE: muscle FLV curve computed by muscle_gain
  """

  FIXED = mujoco.mjtGain.mjGAIN_FIXED
  AFFINE = mujoco.mjtGain.mjGAIN_AFFINE
  MUSCLE = mujoco.mjtGain.mjGAIN_MUSCLE
  # unsupported: USER


class BiasType(enum.IntEnum):
  """Type of actuator bias.

  Members:
    NONE: no bias
    AFFINE: const + kp*length + kv*velocity
    MUSCLE: muscle passive force computed by muscle_bias
  """

  NONE = mujoco.mjtBias.mjBIAS_NONE
  AFFINE = mujoco.mjtBias.mjBIAS_AFFINE
  MUSCLE = mujoco.mjtBias.mjBIAS_MUSCLE
  # unsupported: USER


class JointType(enum.IntEnum):
  """Type of degree of freedom.

  Members:
    FREE:  global position and orientation (quat)       (7,)
    BALL:  orientation (quat) relative to parent        (4,)
    SLIDE: sliding distance along body-fixed axis       (1,)
    HINGE: rotation angle (rad) around body-fixed axis  (1,)
  """

  FREE = mujoco.mjtJoint.mjJNT_FREE
  BALL = mujoco.mjtJoint.mjJNT_BALL
  SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
  HINGE = mujoco.mjtJoint.mjJNT_HINGE

  def dof_width(self) -> int:
    return {0: 6, 1: 3, 2: 1, 3: 1}[self.value]

  def qpos_width(self) -> int:
    return {0: 7, 1: 4, 2: 1, 3: 1}[self.value]


class ConeType(enum.IntEnum):
  """Type of friction cone.

  Members:
    PYRAMIDAL: pyramidal
    ELLIPTIC: elliptic
  """

  PYRAMIDAL = mujoco.mjtCone.mjCONE_PYRAMIDAL
  ELLIPTIC = mujoco.mjtCone.mjCONE_ELLIPTIC


class GeomType(enum.IntEnum):
  """Type of geometry.

  Members:
    PLANE: plane
    HFIELD: height field
    SPHERE: sphere
    CAPSULE: capsule
    ELLIPSOID: ellipsoid
    CYLINDER: cylinder
    BOX: box
    MESH: mesh
  """

  PLANE = mujoco.mjtGeom.mjGEOM_PLANE
  HFIELD = mujoco.mjtGeom.mjGEOM_HFIELD
  SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
  CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
  ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
  CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
  BOX = mujoco.mjtGeom.mjGEOM_BOX
  MESH = mujoco.mjtGeom.mjGEOM_MESH
  # unsupported: NGEOMTYPES, ARROW*, LINE, SKIN, LABEL, NONE


NUM_GEOM_TYPES = 8


class vec5f(wp.types.vector(length=5, dtype=wp.float32)):
  pass


class vec10f(wp.types.vector(length=10, dtype=wp.float32)):
  pass


vec5 = vec5f
vec10 = vec10f
array2df = wp.array2d(dtype=wp.float32)
array3df = wp.array3d(dtype=wp.float32)


@wp.struct
class Option:
  """Physics options.

  Attributes:
    cone: type of friction cone
    disableflags: bit flags for disabling standard features
    gravity: gravitational acceleration (3,)
    impratio: ratio of friction-to-normal contact impedance
    integrator: integration mode
    is_sparse: whether to use sparse representations (warp only)
    iterations: number of main solver iterations
    ls_iterations: maximum number of CG/Newton linesearch iterations
    ls_tolerance: CG/Newton linesearch tolerance
    solver: solver algorithm
    timestep: timestep
    tolerance: main solver tolerance
  """
  cone: int  # mjtCone
  disableflags: int
  gravity: wp.vec3
  impratio: wp.float32
  integrator: int  # mjtIntegrator
  is_sparse: bool  # warp only
  iterations: int
  ls_iterations: int
  ls_tolerance: float
  solver: int  # mjtSolver
  timestep: float
  tolerance: float


@wp.struct
class Statistic:
  """Model statistics (in qpos0).

  Attributes:
    meaninertia: mean diagonal inertia
  """
  meaninertia: float


@wp.struct
class Constraint:
  # efc
  J: wp.array(dtype=wp.float32, ndim=2)
  D: wp.array(dtype=wp.float32, ndim=1)
  pos: wp.array(dtype=wp.float32, ndim=1)
  aref: wp.array(dtype=wp.float32, ndim=1)
  force: wp.array(dtype=wp.float32, ndim=1)
  margin: wp.array(dtype=wp.float32, ndim=1)
  worldid: wp.array(dtype=wp.int32, ndim=1)  # warp only
  # solver context
  Jaref: wp.array(dtype=wp.float32, ndim=1)
  Ma: wp.array(dtype=wp.float32, ndim=2)
  grad: wp.array(dtype=wp.float32, ndim=2)
  grad_dot: wp.array(dtype=wp.float32, ndim=1)
  Mgrad: wp.array(dtype=wp.float32, ndim=2)
  search: wp.array(dtype=wp.float32, ndim=2)
  search_dot: wp.array(dtype=wp.float32, ndim=1)
  gauss: wp.array(dtype=wp.float32, ndim=1)
  cost: wp.array(dtype=wp.float32, ndim=1)
  prev_cost: wp.array(dtype=wp.float32, ndim=1)
  solver_niter: wp.array(dtype=wp.int32, ndim=1)
  active: wp.array(dtype=wp.int32, ndim=1)
  gtol: wp.array(dtype=wp.float32, ndim=1)
  mv: wp.array(dtype=wp.float32, ndim=2)
  jv: wp.array(dtype=wp.float32, ndim=1)
  quad: wp.array(dtype=wp.vec3f, ndim=1)
  quad_gauss: wp.array(dtype=wp.vec3f, ndim=1)
  h: wp.array(dtype=wp.float32, ndim=3)
  alpha: wp.array(dtype=wp.float32, ndim=1)
  prev_grad: wp.array(dtype=wp.float32, ndim=2)
  prev_Mgrad: wp.array(dtype=wp.float32, ndim=2)
  beta: wp.array(dtype=wp.float32, ndim=1)
  beta_num: wp.array(dtype=wp.float32, ndim=1)
  beta_den: wp.array(dtype=wp.float32, ndim=1)
  done: wp.array(dtype=wp.int32, ndim=1)
  # linesearch
  ls_done: wp.array(dtype=bool, ndim=1)
  p0: wp.array(dtype=wp.vec3, ndim=1)
  lo: wp.array(dtype=wp.vec3, ndim=1)
  lo_alpha: wp.array(dtype=wp.float32, ndim=1)
  hi: wp.array(dtype=wp.vec3, ndim=1)
  hi_alpha: wp.array(dtype=wp.float32, ndim=1)
  lo_next: wp.array(dtype=wp.vec3, ndim=1)
  lo_next_alpha: wp.array(dtype=wp.float32, ndim=1)
  hi_next: wp.array(dtype=wp.vec3, ndim=1)
  hi_next_alpha: wp.array(dtype=wp.float32, ndim=1)
  mid: wp.array(dtype=wp.vec3, ndim=1)
  mid_alpha: wp.array(dtype=wp.float32, ndim=1)


@wp.struct
class Model:
  nq: int
  nv: int
  na: int
  nu: int
  nbody: int
  njnt: int
  ngeom: int
  nsite: int
  nmocap: int
  nM: int
  nexclude: int
  opt: Option
  stat: Statistic
  qpos0: wp.array(dtype=wp.float32, ndim=1)
  qpos_spring: wp.array(dtype=wp.float32, ndim=1)
  body_tree: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_treeadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_moment_offset_nv: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_moment_offset_nu: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_moment_tileadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_moment_tilesize_nv: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_moment_tilesize_nu: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qM_fullm_i: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qM_fullm_j: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qM_mulm_i: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qM_mulm_j: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qM_madr_ij: wp.array(dtype=wp.int32, ndim=1)  # warp only
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
  body_weldid: wp.array(dtype=wp.int32, ndim=1)
  body_pos: wp.array(dtype=wp.vec3, ndim=1)
  body_quat: wp.array(dtype=wp.quat, ndim=1)
  body_ipos: wp.array(dtype=wp.vec3, ndim=1)
  body_iquat: wp.array(dtype=wp.quat, ndim=1)
  body_rootid: wp.array(dtype=wp.int32, ndim=1)
  body_inertia: wp.array(dtype=wp.vec3, ndim=1)
  body_mass: wp.array(dtype=wp.float32, ndim=1)
  subtree_mass: wp.array(dtype=wp.float32, ndim=1)
  body_invweight0: wp.array(dtype=wp.float32, ndim=2)
  body_geomnum: wp.array(dtype=wp.int32, ndim=1)
  body_geomadr: wp.array(dtype=wp.int32, ndim=1)
  body_contype: wp.array(dtype=wp.int32, ndim=1)
  body_conaffinity: wp.array(dtype=wp.int32, ndim=1)
  jnt_bodyid: wp.array(dtype=wp.int32, ndim=1)
  jnt_limited: wp.array(dtype=wp.int32, ndim=1)
  jnt_limited_slide_hinge_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  jnt_solref: wp.array(dtype=wp.vec2, ndim=1)
  jnt_solimp: wp.array(dtype=vec5, ndim=1)
  jnt_type: wp.array(dtype=wp.int32, ndim=1)
  jnt_qposadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_dofadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_axis: wp.array(dtype=wp.vec3, ndim=1)
  jnt_pos: wp.array(dtype=wp.vec3, ndim=1)
  jnt_range: wp.array(dtype=wp.float32, ndim=2)
  jnt_margin: wp.array(dtype=wp.float32, ndim=1)
  jnt_stiffness: wp.array(dtype=wp.float32, ndim=1)
  jnt_actfrclimited: wp.array(dtype=wp.bool, ndim=1)
  jnt_actfrcrange: wp.array(dtype=wp.vec2, ndim=1)
  geom_type: wp.array(dtype=wp.int32, ndim=1)
  geom_bodyid: wp.array(dtype=wp.int32, ndim=1)
  geom_conaffinity: wp.array(dtype=wp.int32, ndim=1)
  geom_contype: wp.array(dtype=wp.int32, ndim=1)
  geom_condim: wp.array(dtype=wp.int32, ndim=1)
  geom_pos: wp.array(dtype=wp.vec3, ndim=1)
  geom_quat: wp.array(dtype=wp.quat, ndim=1)
  geom_size: wp.array(dtype=wp.vec3, ndim=1)
  geom_priority: wp.array(dtype=wp.int32, ndim=1)
  geom_solmix: wp.array(dtype=wp.float32, ndim=1)
  geom_solref: wp.array(dtype=wp.vec2, ndim=1)
  geom_solimp: wp.array(dtype=vec5, ndim=1)
  geom_friction: wp.array(dtype=wp.vec3, ndim=1)
  geom_margin: wp.array(dtype=wp.float32, ndim=1)
  geom_gap: wp.array(dtype=wp.float32, ndim=1)
  geom_rbound: wp.array(dtype=wp.float32, ndim=1)
  geom_aabb: wp.array(dtype=wp.vec3, ndim=2)
  geom_dataid: wp.array(dtype=wp.int32, ndim=1)
  mesh_vertadr: wp.array(dtype=wp.int32, ndim=1)
  mesh_vertnum: wp.array(dtype=wp.int32, ndim=1)
  mesh_vert: wp.array(dtype=wp.vec3, ndim=1)
  site_pos: wp.array(dtype=wp.vec3, ndim=1)
  site_quat: wp.array(dtype=wp.quat, ndim=1)
  site_bodyid: wp.array(dtype=wp.int32, ndim=1)
  dof_bodyid: wp.array(dtype=wp.int32, ndim=1)
  dof_jntid: wp.array(dtype=wp.int32, ndim=1)
  dof_parentid: wp.array(dtype=wp.int32, ndim=1)
  dof_Madr: wp.array(dtype=wp.int32, ndim=1)
  dof_armature: wp.array(dtype=wp.float32, ndim=1)
  dof_invweight0: wp.array(dtype=wp.float32, ndim=1)
  dof_damping: wp.array(dtype=wp.float32, ndim=1)
  dof_tri_row: wp.array(dtype=wp.int32, ndim=1)  # warp only
  dof_tri_col: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_trntype: wp.array(dtype=wp.int32, ndim=1)
  actuator_trnid: wp.array(dtype=wp.int32, ndim=2)
  actuator_ctrllimited: wp.array(dtype=wp.bool, ndim=1)
  actuator_ctrlrange: wp.array(dtype=wp.vec2, ndim=1)
  actuator_forcelimited: wp.array(dtype=wp.bool, ndim=1)
  actuator_forcerange: wp.array(dtype=wp.vec2, ndim=1)
  actuator_gaintype: wp.array(dtype=wp.int32, ndim=1)
  actuator_gainprm: wp.array(dtype=wp.float32, ndim=2)
  actuator_biasprm: wp.array(dtype=wp.float32, ndim=2)
  actuator_gear: wp.array(dtype=wp.spatial_vector, ndim=1)
  actuator_actlimited: wp.array(dtype=wp.bool, ndim=1)
  actuator_actrange: wp.array(dtype=wp.vec2, ndim=1)
  actuator_actadr: wp.array(dtype=wp.int32, ndim=1)
  actuator_biastype: wp.array(dtype=wp.int32, ndim=1)
  actuator_dyntype: wp.array(dtype=wp.int32, ndim=1)
  actuator_dynprm: wp.array(dtype=vec10f, ndim=1)
  exclude_signature: wp.array(dtype=wp.int32, ndim=1)
  actuator_affine_bias_gain: bool

  
@wp.struct
class Contact:
  """Contact data.

  Attributes:
    dist: distance between nearest points; neg: penetration
    pos: position of contact point: midpoint between geoms
    frame: normal is in [0-2], points from geom[0] to geom[1]
    includemargin: include if dist<includemargin=margin-gap
    friction: tangent1, 2, spin, roll1, 2
    solref: constraint solver reference, normal direction
    solreffriction: constraint solver reference, friction directions
    solimp: constraint solver impedance
    dim: contact space dimensionality: 1, 3, 4 or 6
    geom: geom ids; -1 for flex
    efc_address: address in efc; -1: not included
    worldid: world id (warp only)
  """
  dist: wp.array(dtype=wp.float32, ndim=1)
  pos: wp.array(dtype=wp.vec3f, ndim=1)
  frame: wp.array(dtype=wp.mat33f, ndim=1)
  includemargin: wp.array(dtype=wp.float32, ndim=1)
  friction: wp.array(dtype=vec5, ndim=1)
  solref: wp.array(dtype=wp.vec2f, ndim=1)
  solreffriction: wp.array(dtype=wp.vec2f, ndim=1)
  solimp: wp.array(dtype=vec5, ndim=1)
  dim: wp.array(dtype=wp.int32, ndim=1)
  geom: wp.array(dtype=wp.vec2i, ndim=1)
  efc_address: wp.array(dtype=wp.int32, ndim=1)
  worldid: wp.array(dtype=wp.int32, ndim=1)


@wp.struct
class Data:
  """ Data. """

  ## VARIABLE sizes
  # number of detected contacts
  ncon: wp.array(dtype=wp.int32, ndim=1)
  # number of limit constraints
  nl: int
  # number of constraints
  nefc: wp.array(dtype=wp.int32, ndim=1)

  ## GLOBAL properties
  # simulation time
  time: float

  ## STATE
  # position (nq x 1)
  qpos: wp.array(dtype=wp.float32, ndim=2)
  # velocity (nv x 1)
  qvel: wp.array(dtype=wp.float32, ndim=2)
  # actuator activation (na x 1)
  act: wp.array(dtype=wp.float32, ndim=2)
  # acceleration used for warmstart (nv x 1)
  qacc_warmstart: wp.array(dtype=wp.float32, ndim=2)

  ## CONTROL
  # control (nu x 1)
  ctrl: wp.array(dtype=wp.float32, ndim=2)
  # applied generalized force (nv x 1)
  qfrc_applied: wp.array(dtype=wp.float32, ndim=2)
  # applied Cartesian force/torque (nbody x 6)
  xfrc_applied: wp.array(dtype=wp.spatial_vector, ndim=2)

  ## MOCAP DATA
  # position of mocap bodies (nmocap x 3)
  mocap_pos: wp.array(dtype=wp.vec3, ndim=2)
  # orientation of mocap bodies (nmocap x 4)
  mocap_quat: wp.array(dtype=wp.quat, ndim=2)

  ## DYNAMICS
  # acceleration (nv x 1) 
  qacc: wp.array(dtype=wp.float32, ndim=2)
  # time-derivative of actuator activation (na x 1)
  act_dot: wp.array(dtype=wp.float32, ndim=2)

  ##-------------------- POSITION dependent
  
  ## COMPUTED by mj_fwdPosition/mj_kinematics
  # Cartesian position of body frame (nbody x 3)
  xpos: wp.array(dtype=wp.vec3, ndim=2)
  # Cartesian orientation of body frame (nbody x 4)
  xquat: wp.array(dtype=wp.quat, ndim=2)
  # Cartesian orientation of body frame (nbody x 9)
  xmat: wp.array(dtype=wp.mat33, ndim=2)
  # Cartesian position of body com (nbody x 3)
  xipos: wp.array(dtype=wp.vec3, ndim=2)
  # Cartesian orientation of body inertia (nbody x 9)
  ximat: wp.array(dtype=wp.mat33, ndim=2)
  # Cartesian position of joint anchor (njnt x 3)
  xanchor: wp.array(dtype=wp.vec3, ndim=2)
  # Cartesian joint axis (njnt x 3)
  xaxis: wp.array(dtype=wp.vec3, ndim=2)
  # Cartesian geom position (ngeom x 3)
  geom_xpos: wp.array(dtype=wp.vec3, ndim=2)
  # Cartesian geom orientation (ngeom x 9)
  geom_xmat: wp.array(dtype=wp.mat33, ndim=2)
  # Cartesian site position (nsite x 3)
  site_xpos: wp.array(dtype=wp.vec3, ndim=2)
  # Cartesian site orientation (nsite x 9)
  site_xmat: wp.array(dtype=wp.mat33, ndim=2)

  ## COMPUTED by fwd_position/com_pos
  # center of mass of each subtree (nbody x 3)
  subtree_com: wp.array(dtype=wp.vec3, ndim=2)
  # com-based motion axis of each dof (rot:lin) (nv x 6)
  cdof: wp.array(dtype=wp.spatial_vector, ndim=2)
  # com-based body inertia and mass (nbody x 10)
  cinert: wp.array(dtype=vec10, ndim=2)

  ## COMPUTED by fwd_position/transmission
  # actuator lengths (nu x 1)
  actuator_length: wp.array(dtype=wp.float32, ndim=2)
  # actuator moments (nJmom x 1)
  actuator_moment: wp.array(dtype=wp.float32, ndim=3)

  ## COMPUTED by fwd_position/crb
  # com-based composite inertia and mass (nbody x 10)
  crb: wp.array(dtype=vec10, ndim=2)
  # total inertia (sparse) (nM x 1)
  qM: wp.array(dtype=wp.float32, ndim=3)

  ## COMPUTED by fwd_position/factor_m
  # L'*D*L factorization of M (sparse) (nM x 1)
  qLD: wp.array(dtype=wp.float32, ndim=3)
  # 1/diag(D) (nv x 1)
  qLDiagInv: wp.array(dtype=wp.float32, ndim=2)

  ##-------------------- POSITION, VELOCITY dependent
  ## COMPUTED by fwd_velocity
  # actuator velocities (nu x 1)
  actuator_velocity: wp.array(dtype=wp.float32, ndim=2)
  
  ## COMPUTED by fwd_velocity/com_vel
  # com-based velocity (rot:lin) (nbody x 6)
  cvel: wp.array(dtype=wp.spatial_vector, ndim=2)
  # time-derivative of cdof (rot:lin) (nv x 6)
  cdof_dot: wp.array(dtype=wp.spatial_vector, ndim=2)
  
  ## COMPUTED by fwd_velocity/rne (without acceleration)
  # C(qpos,qvel) (nv x 1)
  qfrc_bias: wp.array(dtype=wp.float32, ndim=2)

  ## COMPUTED by fwd_velocity/passive
  # passive spring force (nv x 1)
  qfrc_spring: wp.array(dtype=wp.float32, ndim=2)
  # passive damper force (nv x 1)
  qfrc_damper: wp.array(dtype=wp.float32, ndim=2)
  # total passive force (nv x 1)
  qfrc_passive: wp.array(dtype=wp.float32, ndim=2)

  ## -------------------- POSITION, VELOCITY, CONTROL/ACCELERATION dependent
  
  ## COMPUTED by fwd_actuation
  # actuator force in actuation space (nu x 1)
  actuator_force: wp.array(dtype=wp.float32, ndim=2)
  # actuator force (nv x 1)
  qfrc_actuator: wp.array(dtype=wp.float32, ndim=2)

  ## COMPUTED by fwd_acceleration
  # net unconstrained force (nv x 1)
  qfrc_smooth: wp.array(dtype=wp.float32, ndim=2)
  # unconstrained acceleration (nv x 1)
  qacc_smooth: wp.array(dtype=wp.float32, ndim=2)

  ## COMPUTED by _qfrc_constraint
  # constraint force (nv x 1)
  qfrc_constraint: wp.array(dtype=wp.float32, ndim=2)

  ##-------------------- arena-allocated: POSITION dependent

  ## COMPUTED by collision
  # array of all detected contacts (ncon x 1)
  contact: Contact
  
  ## COMPUTED by make_constraint
  # constraint data
  efc: Constraint

  ##-------------------- Warp only
  # sizes
  nworld: int
  nconmax: int
  nefc_total: wp.array(dtype=wp.int32, ndim=1)  # warp only
  njmax: int

  # applied forces
  qfrc_applied: wp.array(dtype=wp.float32, ndim=2)
  xfrc_applied: wp.array(dtype=wp.spatial_vector, ndim=2)

  # arrays used for smooth.rne
  rne_cacc: wp.array(dtype=wp.spatial_vector, ndim=2)
  rne_cfrc: wp.array(dtype=wp.spatial_vector, ndim=2)

  # temp arrays
  qfrc_integration: wp.array(dtype=wp.float32, ndim=2)
  qacc_integration: wp.array(dtype=wp.float32, ndim=2)
  act_vel_integration: wp.array(dtype=wp.float32, ndim=2)

  qM_integration: wp.array(dtype=wp.float32, ndim=3)
  qLD_integration: wp.array(dtype=wp.float32, ndim=3)
  qLDiagInv_integration: wp.array(dtype=wp.float32, ndim=2)

  # sweep and prune broadphase arrays
  boxes_sorted: wp.array(dtype=wp.vec3, ndim=3)
  box_projections_lower: wp.array(dtype=wp.float32, ndim=2)
  box_projections_upper: wp.array(dtype=wp.float32, ndim=2)
  box_sorting_indexer: wp.array(dtype=wp.int32, ndim=2)
  ranges: wp.array(dtype=wp.int32, ndim=2)
  cumulative_sum: wp.array(dtype=wp.int32, ndim=1)
  segment_indices: wp.array(dtype=wp.int32, ndim=1)
  dyn_geom_aabb: wp.array(dtype=wp.vec3, ndim=3)

  # collision driver
  collision_pair: wp.array(dtype=wp.vec2i, ndim=1)
  collision_type: wp.array(dtype=wp.int32, ndim=1)
  collision_worldid: wp.array(dtype=wp.int32, ndim=1)
  ncollision: wp.array(dtype=wp.int32, ndim=1)
