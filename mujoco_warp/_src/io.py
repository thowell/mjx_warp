# Copyright 2025 The Newton Developers
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

import mujoco
import numpy as np
import warp as wp

from . import support
from . import types


def put_model(mjm: mujoco.MjModel) -> types.Model:
  m = types.Model()

  # TODO(team): check supported constraints
  
  m.nq = mjm.nq
  m.nv = mjm.nv
  m.na = mjm.na
  m.nu = mjm.nu
  m.nbody = mjm.nbody
  m.njnt = mjm.njnt
  m.ngeom = mjm.ngeom
  m.nsite = mjm.nsite
  m.nmocap = mjm.nmocap
  m.nM = mjm.nM
  m.nexclude = mjm.nexclude
  m.opt.timestep = mjm.opt.timestep
  m.opt.tolerance = mjm.opt.tolerance
  m.opt.ls_tolerance = mjm.opt.ls_tolerance
  m.opt.gravity = wp.vec3(mjm.opt.gravity)
  m.opt.cone = mjm.opt.cone
  m.opt.solver = mjm.opt.solver
  m.opt.iterations = mjm.opt.iterations
  m.opt.ls_iterations = mjm.opt.ls_iterations
  m.opt.integrator = mjm.opt.integrator
  m.opt.disableflags = mjm.opt.disableflags
  m.opt.impratio = wp.float32(mjm.opt.impratio)
  m.opt.is_sparse = support.is_sparse(mjm)
  m.stat.meaninertia = mjm.stat.meaninertia

  m.qpos0 = wp.array(mjm.qpos0, dtype=wp.float32, ndim=1)
  m.qpos_spring = wp.array(mjm.qpos_spring, dtype=wp.float32, ndim=1)

  # dof lower triangle row and column indices
  dof_tri_row, dof_tri_col = np.tril_indices(mjm.nv)

  # indices for sparse qM full_m
  is_, js = [], []
  for i in range(mjm.nv):
    j = i
    while j > -1:
      is_.append(i)
      js.append(j)
      j = mjm.dof_parentid[j]
  qM_fullm_i = is_
  qM_fullm_j = js

  # indices for sparse qM mul_m
  is_, js, madr_ijs = [], [], []
  for i in range(mjm.nv):
    madr_ij, j = mjm.dof_Madr[i], i

    while True:
      madr_ij, j = madr_ij + 1, mjm.dof_parentid[j]
      if j == -1:
        break
      is_, js, madr_ijs = is_ + [i], js + [j], madr_ijs + [madr_ij]

  qM_mulm_i, qM_mulm_j, qM_madr_ij = (
    np.array(x, dtype=np.int32) for x in (is_, js, madr_ijs)
  )

  jnt_limited_slide_hinge_adr = np.nonzero(
    mjm.jnt_limited
    & (
      (mjm.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE)
      | (mjm.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    )
  )[0]

  # body_tree is BFS ordering of body ids
  # body_treeadr contains starting index of each body tree level
  bodies, body_depth = {}, np.zeros(mjm.nbody, dtype=int) - 1
  for i in range(mjm.nbody):
    body_depth[i] = body_depth[mjm.body_parentid[i]] + 1
    bodies.setdefault(body_depth[i], []).append(i)
  body_tree = np.concatenate([bodies[i] for i in range(len(bodies))])
  tree_off = [0] + [len(bodies[i]) for i in range(len(bodies))]
  body_treeadr = np.cumsum(tree_off)[:-1]

  m.body_tree = wp.array(body_tree, dtype=wp.int32, ndim=1)
  m.body_treeadr = wp.array(body_treeadr, dtype=wp.int32, ndim=1, device="cpu")

  qLD_update_tree = np.empty(shape=(0, 3), dtype=int)
  qLD_update_treeadr = np.empty(shape=(0,), dtype=int)
  qLD_tile = np.empty(shape=(0,), dtype=int)
  qLD_tileadr = np.empty(shape=(0,), dtype=int)
  qLD_tilesize = np.empty(shape=(0,), dtype=int)

  if support.is_sparse(mjm):
    # qLD_update_tree has dof tree ordering of qLD updates for sparse factor m
    # qLD_update_treeadr contains starting index of each dof tree level
    qLD_updates, dof_depth = {}, np.zeros(mjm.nv, dtype=int) - 1
    for k in range(mjm.nv):
      dof_depth[k] = dof_depth[mjm.dof_parentid[k]] + 1
      i = mjm.dof_parentid[k]
      Madr_ki = mjm.dof_Madr[k] + 1
      while i > -1:
        qLD_updates.setdefault(dof_depth[i], []).append((i, k, Madr_ki))
        i = mjm.dof_parentid[i]
        Madr_ki += 1

    # qLD_treeadr contains starting indicies of each level of sparse updates
    qLD_update_tree = np.concatenate([qLD_updates[i] for i in range(len(qLD_updates))])
    tree_off = [0] + [len(qLD_updates[i]) for i in range(len(qLD_updates))]
    qLD_update_treeadr = np.cumsum(tree_off)[:-1]
  else:
    # qLD_tile has the dof id of each tile in qLD for dense factor m
    # qLD_tileadr contains starting index in qLD_tile of each tile group
    # qLD_tilesize has the square tile size of each tile group
    tile_corners = [i for i in range(mjm.nv) if mjm.dof_parentid[i] == -1]
    tiles = {}
    for i in range(len(tile_corners)):
      tile_beg = tile_corners[i]
      tile_end = mjm.nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
      tiles.setdefault(tile_end - tile_beg, []).append(tile_beg)
    qLD_tile = np.concatenate([tiles[sz] for sz in sorted(tiles.keys())])
    tile_off = [0] + [len(tiles[sz]) for sz in sorted(tiles.keys())]
    qLD_tileadr = np.cumsum(tile_off)[:-1]
    qLD_tilesize = np.array(sorted(tiles.keys()))

  # tiles for actuator_moment - needs nu + nv tile size and offset
  actuator_moment_offset_nv = np.empty(shape=(0,), dtype=int)
  actuator_moment_offset_nu = np.empty(shape=(0,), dtype=int)
  actuator_moment_tileadr = np.empty(shape=(0,), dtype=int)
  actuator_moment_tilesize_nv = np.empty(shape=(0,), dtype=int)
  actuator_moment_tilesize_nu = np.empty(shape=(0,), dtype=int)

  if not support.is_sparse(mjm):
    # how many actuators for each tree
    tile_corners = [i for i in range(mjm.nv) if mjm.dof_parentid[i] == -1]
    tree_id = mjm.dof_treeid[tile_corners]
    num_trees = int(np.max(tree_id))
    tree = mjm.body_treeid[mjm.jnt_bodyid[mjm.actuator_trnid[:, 0]]]
    counts, ids = np.histogram(tree, bins=np.arange(0, num_trees + 2))
    acts_per_tree = dict(zip([int(i) for i in ids], [int(i) for i in counts]))

    tiles = {}
    act_beg = 0
    for i in range(len(tile_corners)):
      tile_beg = tile_corners[i]
      tile_end = mjm.nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
      tree = int(tree_id[i])
      act_num = acts_per_tree[tree]
      tiles.setdefault((tile_end - tile_beg, act_num), []).append((tile_beg, act_beg))
      act_beg += act_num

    sorted_keys = sorted(tiles.keys())
    actuator_moment_offset_nv = [
      t[0] for key in sorted_keys for t in tiles.get(key, [])
    ]
    actuator_moment_offset_nu = [
      t[1] for key in sorted_keys for t in tiles.get(key, [])
    ]
    tile_off = [0] + [len(tiles[sz]) for sz in sorted(tiles.keys())]
    actuator_moment_tileadr = np.cumsum(tile_off)[:-1]  # offset
    actuator_moment_tilesize_nv = np.array(
      [a[0] for a in sorted_keys]
    )  # for this level
    actuator_moment_tilesize_nu = np.array(
      [int(a[1]) for a in sorted_keys]
    )  # for this level

  m.qM_fullm_i = wp.array(qM_fullm_i, dtype=wp.int32, ndim=1)
  m.qM_fullm_j = wp.array(qM_fullm_j, dtype=wp.int32, ndim=1)
  m.qM_mulm_i = wp.array(qM_mulm_i, dtype=wp.int32, ndim=1)
  m.qM_mulm_j = wp.array(qM_mulm_j, dtype=wp.int32, ndim=1)
  m.qM_madr_ij = wp.array(qM_madr_ij, dtype=wp.int32, ndim=1)
  m.qLD_update_tree = wp.array(qLD_update_tree, dtype=wp.vec3i, ndim=1)
  m.qLD_update_treeadr = wp.array(
    qLD_update_treeadr, dtype=wp.int32, ndim=1, device="cpu"
  )
  m.qLD_tile = wp.array(qLD_tile, dtype=wp.int32, ndim=1)
  m.qLD_tileadr = wp.array(qLD_tileadr, dtype=wp.int32, ndim=1, device="cpu")
  m.qLD_tilesize = wp.array(qLD_tilesize, dtype=wp.int32, ndim=1, device="cpu")
  m.actuator_moment_offset_nv = wp.array(
    actuator_moment_offset_nv, dtype=wp.int32, ndim=1
  )
  m.actuator_moment_offset_nu = wp.array(
    actuator_moment_offset_nu, dtype=wp.int32, ndim=1
  )
  m.actuator_moment_tileadr = wp.array(
    actuator_moment_tileadr, dtype=wp.int32, ndim=1, device="cpu"
  )
  m.actuator_moment_tilesize_nv = wp.array(
    actuator_moment_tilesize_nv, dtype=wp.int32, ndim=1, device="cpu"
  )
  m.actuator_moment_tilesize_nu = wp.array(
    actuator_moment_tilesize_nu, dtype=wp.int32, ndim=1, device="cpu"
  )
  m.body_dofadr = wp.array(mjm.body_dofadr, dtype=wp.int32, ndim=1)
  m.body_dofnum = wp.array(mjm.body_dofnum, dtype=wp.int32, ndim=1)
  m.body_jntadr = wp.array(mjm.body_jntadr, dtype=wp.int32, ndim=1)
  m.body_jntnum = wp.array(mjm.body_jntnum, dtype=wp.int32, ndim=1)
  m.body_parentid = wp.array(mjm.body_parentid, dtype=wp.int32, ndim=1)
  m.body_mocapid = wp.array(mjm.body_mocapid, dtype=wp.int32, ndim=1)
  m.body_weldid = wp.array(mjm.body_weldid, dtype=wp.int32, ndim=1)
  m.body_pos = wp.array(mjm.body_pos, dtype=wp.vec3, ndim=1)
  m.body_quat = wp.array(mjm.body_quat, dtype=wp.quat, ndim=1)
  m.body_ipos = wp.array(mjm.body_ipos, dtype=wp.vec3, ndim=1)
  m.body_iquat = wp.array(mjm.body_iquat, dtype=wp.quat, ndim=1)
  m.body_rootid = wp.array(mjm.body_rootid, dtype=wp.int32, ndim=1)
  m.body_inertia = wp.array(mjm.body_inertia, dtype=wp.vec3, ndim=1)
  m.body_mass = wp.array(mjm.body_mass, dtype=wp.float32, ndim=1)

  subtree_mass = np.copy(mjm.body_mass)
  # TODO(team): should this be [mjm.nbody - 1, 0) ?
  for i in range(mjm.nbody - 1, -1, -1):
    subtree_mass[mjm.body_parentid[i]] += subtree_mass[i]

  m.subtree_mass = wp.array(subtree_mass, dtype=wp.float32, ndim=1)
  m.body_invweight0 = wp.array(mjm.body_invweight0, dtype=wp.float32, ndim=2)
  m.body_geomnum = wp.array(mjm.body_geomnum, dtype=wp.int32, ndim=1)
  m.body_geomadr = wp.array(mjm.body_geomadr, dtype=wp.int32, ndim=1)
  m.body_contype = wp.array(mjm.body_contype, dtype=wp.int32, ndim=1)
  m.body_conaffinity = wp.array(mjm.body_conaffinity, dtype=wp.int32, ndim=1)
  m.jnt_bodyid = wp.array(mjm.jnt_bodyid, dtype=wp.int32, ndim=1)
  m.jnt_limited = wp.array(mjm.jnt_limited, dtype=wp.int32, ndim=1)
  m.jnt_limited_slide_hinge_adr = wp.array(
    jnt_limited_slide_hinge_adr, dtype=wp.int32, ndim=1
  )
  m.jnt_type = wp.array(mjm.jnt_type, dtype=wp.int32, ndim=1)
  m.jnt_solref = wp.array(mjm.jnt_solref, dtype=wp.vec2f, ndim=1)
  m.jnt_solimp = wp.array(mjm.jnt_solimp, dtype=types.vec5, ndim=1)
  m.jnt_qposadr = wp.array(mjm.jnt_qposadr, dtype=wp.int32, ndim=1)
  m.jnt_dofadr = wp.array(mjm.jnt_dofadr, dtype=wp.int32, ndim=1)
  m.jnt_axis = wp.array(mjm.jnt_axis, dtype=wp.vec3, ndim=1)
  m.jnt_pos = wp.array(mjm.jnt_pos, dtype=wp.vec3, ndim=1)
  m.jnt_range = wp.array(mjm.jnt_range, dtype=wp.float32, ndim=2)
  m.jnt_margin = wp.array(mjm.jnt_margin, dtype=wp.float32, ndim=1)
  m.jnt_stiffness = wp.array(mjm.jnt_stiffness, dtype=wp.float32, ndim=1)
  m.jnt_actfrclimited = wp.array(mjm.jnt_actfrclimited, dtype=wp.bool, ndim=1)
  m.jnt_actfrcrange = wp.array(mjm.jnt_actfrcrange, dtype=wp.vec2, ndim=1)
  m.geom_type = wp.array(mjm.geom_type, dtype=wp.int32, ndim=1)
  m.geom_bodyid = wp.array(mjm.geom_bodyid, dtype=wp.int32, ndim=1)
  m.geom_conaffinity = wp.array(mjm.geom_conaffinity, dtype=wp.int32, ndim=1)
  m.geom_contype = wp.array(mjm.geom_contype, dtype=wp.int32, ndim=1)
  m.geom_condim = wp.array(mjm.geom_condim, dtype=wp.int32, ndim=1)
  m.geom_pos = wp.array(mjm.geom_pos, dtype=wp.vec3, ndim=1)
  m.geom_quat = wp.array(mjm.geom_quat, dtype=wp.quat, ndim=1)
  m.geom_size = wp.array(mjm.geom_size, dtype=wp.vec3, ndim=1)
  m.geom_priority = wp.array(mjm.geom_priority, dtype=wp.int32, ndim=1)
  m.geom_solmix = wp.array(mjm.geom_solmix, dtype=wp.float32, ndim=1)
  m.geom_solref = wp.array(mjm.geom_solref, dtype=wp.vec2, ndim=1)
  m.geom_solimp = wp.array(mjm.geom_solimp, dtype=types.vec5, ndim=1)
  m.geom_friction = wp.array(mjm.geom_friction, dtype=wp.vec3, ndim=1)
  m.geom_margin = wp.array(mjm.geom_margin, dtype=wp.float32, ndim=1)
  m.geom_gap = wp.array(mjm.geom_gap, dtype=wp.float32, ndim=1)
  m.geom_aabb = wp.array(mjm.geom_aabb, dtype=wp.vec3, ndim=3)
  m.geom_rbound = wp.array(mjm.geom_rbound, dtype=wp.float32, ndim=1)
  m.geom_dataid = wp.array(mjm.geom_dataid, dtype=wp.int32, ndim=1)
  m.mesh_vertadr = wp.array(mjm.mesh_vertadr, dtype=wp.int32, ndim=1)
  m.mesh_vertnum = wp.array(mjm.mesh_vertnum, dtype=wp.int32, ndim=1)
  m.mesh_vert = wp.array(mjm.mesh_vert, dtype=wp.vec3, ndim=1)
  m.site_pos = wp.array(mjm.site_pos, dtype=wp.vec3, ndim=1)
  m.site_quat = wp.array(mjm.site_quat, dtype=wp.quat, ndim=1)
  m.site_bodyid = wp.array(mjm.site_bodyid, dtype=wp.int32, ndim=1)
  m.dof_bodyid = wp.array(mjm.dof_bodyid, dtype=wp.int32, ndim=1)
  m.dof_jntid = wp.array(mjm.dof_jntid, dtype=wp.int32, ndim=1)
  m.dof_parentid = wp.array(mjm.dof_parentid, dtype=wp.int32, ndim=1)
  m.dof_Madr = wp.array(mjm.dof_Madr, dtype=wp.int32, ndim=1)
  m.dof_armature = wp.array(mjm.dof_armature, dtype=wp.float32, ndim=1)
  m.dof_damping = wp.array(mjm.dof_damping, dtype=wp.float32, ndim=1)
  m.dof_tri_row = wp.from_numpy(dof_tri_row, dtype=wp.int32)
  m.dof_tri_col = wp.from_numpy(dof_tri_col, dtype=wp.int32)
  m.dof_invweight0 = wp.array(mjm.dof_invweight0, dtype=wp.float32, ndim=1)
  m.actuator_trntype = wp.array(mjm.actuator_trntype, dtype=wp.int32, ndim=1)
  m.actuator_trnid = wp.array(mjm.actuator_trnid, dtype=wp.int32, ndim=2)
  m.actuator_ctrllimited = wp.array(mjm.actuator_ctrllimited, dtype=wp.bool, ndim=1)
  m.actuator_ctrlrange = wp.array(mjm.actuator_ctrlrange, dtype=wp.vec2, ndim=1)
  m.actuator_forcelimited = wp.array(mjm.actuator_forcelimited, dtype=wp.bool, ndim=1)
  m.actuator_forcerange = wp.array(mjm.actuator_forcerange, dtype=wp.vec2, ndim=1)
  m.actuator_gaintype = wp.array(mjm.actuator_gaintype, dtype=wp.int32, ndim=1)
  m.actuator_gainprm = wp.array(mjm.actuator_gainprm, dtype=wp.float32, ndim=2)
  m.actuator_biastype = wp.array(mjm.actuator_biastype, dtype=wp.int32, ndim=1)
  m.actuator_biasprm = wp.array(mjm.actuator_biasprm, dtype=wp.float32, ndim=2)
  m.actuator_gear = wp.array(mjm.actuator_gear, dtype=wp.spatial_vector, ndim=1)
  m.actuator_actlimited = wp.array(mjm.actuator_actlimited, dtype=wp.bool, ndim=1)
  m.actuator_actrange = wp.array(mjm.actuator_actrange, dtype=wp.vec2, ndim=1)
  m.actuator_actadr = wp.array(mjm.actuator_actadr, dtype=wp.int32, ndim=1)
  m.actuator_dyntype = wp.array(mjm.actuator_dyntype, dtype=wp.int32, ndim=1)
  m.actuator_dynprm = wp.array(mjm.actuator_dynprm, dtype=types.vec10f, ndim=1)
  m.exclude_signature = wp.array(mjm.exclude_signature, dtype=wp.int32, ndim=1)

  # short-circuiting here allows us to skip a lot of code in implicit integration
  m.actuator_affine_bias_gain = bool(
    np.any(mjm.actuator_biastype == types.BiasType.AFFINE.value)
    or np.any(mjm.actuator_gaintype == types.GainType.AFFINE.value)
  )

  return m


def _constraint(nv: int, nworld: int, njmax: int) -> types.Constraint:
  efc = types.Constraint()

  efc.J = wp.zeros((njmax, nv), dtype=wp.float32)
  efc.D = wp.zeros((njmax,), dtype=wp.float32)
  efc.pos = wp.zeros((njmax,), dtype=wp.float32)
  efc.aref = wp.zeros((njmax,), dtype=wp.float32)
  efc.force = wp.zeros((njmax,), dtype=wp.float32)
  efc.margin = wp.zeros((njmax,), dtype=wp.float32)
  efc.worldid = wp.zeros((njmax,), dtype=wp.int32)

  efc.Jaref = wp.empty(shape=(njmax,), dtype=wp.float32)
  efc.Ma = wp.empty(shape=(nworld, nv), dtype=wp.float32)
  efc.grad = wp.empty(shape=(nworld, nv), dtype=wp.float32)
  efc.grad_dot = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.Mgrad = wp.empty(shape=(nworld, nv), dtype=wp.float32)
  efc.search = wp.empty(shape=(nworld, nv), dtype=wp.float32)
  efc.search_dot = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.gauss = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.cost = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.prev_cost = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.solver_niter = wp.empty(shape=(nworld,), dtype=wp.int32)
  efc.active = wp.empty(shape=(njmax,), dtype=wp.int32)
  efc.gtol = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.mv = wp.empty(shape=(nworld, nv), dtype=wp.float32)
  efc.jv = wp.empty(shape=(njmax,), dtype=wp.float32)
  efc.quad = wp.empty(shape=(njmax,), dtype=wp.vec3f)
  efc.quad_gauss = wp.empty(shape=(nworld,), dtype=wp.vec3f)
  efc.h = wp.empty(shape=(nworld, nv, nv), dtype=wp.float32)
  efc.alpha = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.prev_grad = wp.empty(shape=(nworld, nv), dtype=wp.float32)
  efc.prev_Mgrad = wp.empty(shape=(nworld, nv), dtype=wp.float32)
  efc.beta = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.beta_num = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.beta_den = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.done = wp.empty(shape=(nworld,), dtype=bool)

  efc.ls_done = wp.zeros(shape=(nworld,), dtype=bool)
  efc.p0 = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.lo = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.lo_alpha = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.hi = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.hi_alpha = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.lo_next = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.lo_next_alpha = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.hi_next = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.hi_next_alpha = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.mid = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.mid_alpha = wp.empty(shape=(nworld,), dtype=wp.float32)

  return efc


def make_data(
  mjm: mujoco.MjModel, nworld: int = 1, nconmax: int = -1, njmax: int = -1
) -> types.Data:
  d = types.Data()
  d.nworld = nworld

  # TODO(team): move to Model?
  if nconmax == -1:
    # TODO(team): heuristic for nconmax
    nconmax = 512
  d.nconmax = nconmax
  if njmax == -1:
    # TODO(team): heuristic for njmax
    njmax = 512
  d.njmax = njmax

  d.ncon = wp.zeros(1, dtype=wp.int32)
  d.nefc = wp.zeros(1, dtype=wp.int32, ndim=1)
  d.nl = 0
  d.time = 0.0

  qpos0 = np.tile(mjm.qpos0, (nworld, 1))
  d.qpos = wp.array(qpos0, dtype=wp.float32, ndim=2)
  d.qvel = wp.zeros((nworld, mjm.nv), dtype=wp.float32, ndim=2)
  d.qacc_warmstart = wp.zeros((nworld, mjm.nv), dtype=wp.float32, ndim=2)
  d.qfrc_applied = wp.zeros((nworld, mjm.nv), dtype=wp.float32, ndim=2)
  d.mocap_pos = wp.zeros((nworld, mjm.nmocap), dtype=wp.vec3)
  d.mocap_quat = wp.zeros((nworld, mjm.nmocap), dtype=wp.quat)
  d.qacc = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.xanchor = wp.zeros((nworld, mjm.njnt), dtype=wp.vec3)
  d.xaxis = wp.zeros((nworld, mjm.njnt), dtype=wp.vec3)
  d.xmat = wp.zeros((nworld, mjm.nbody), dtype=wp.mat33)
  d.xpos = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.xquat = wp.zeros((nworld, mjm.nbody), dtype=wp.quat)
  d.xipos = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.ximat = wp.zeros((nworld, mjm.nbody), dtype=wp.mat33)
  d.subtree_com = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.geom_xpos = wp.zeros((nworld, mjm.ngeom), dtype=wp.vec3)
  d.geom_xmat = wp.zeros((nworld, mjm.ngeom), dtype=wp.mat33)
  d.site_xpos = wp.zeros((nworld, mjm.nsite), dtype=wp.vec3)
  d.site_xmat = wp.zeros((nworld, mjm.nsite), dtype=wp.mat33)
  d.cinert = wp.zeros((nworld, mjm.nbody), dtype=types.vec10)
  d.cdof = wp.zeros((nworld, mjm.nv), dtype=wp.spatial_vector)
  d.ctrl = wp.zeros((nworld, mjm.nu), dtype=wp.float32)
  d.actuator_velocity = wp.zeros((nworld, mjm.nu), dtype=wp.float32)
  d.actuator_force = wp.zeros((nworld, mjm.nu), dtype=wp.float32)
  d.actuator_length = wp.zeros((nworld, mjm.nu), dtype=wp.float32)
  d.actuator_moment = wp.zeros((nworld, mjm.nu, mjm.nv), dtype=wp.float32)
  d.crb = wp.zeros((nworld, mjm.nbody), dtype=types.vec10)
  if support.is_sparse(mjm):
    d.qM = wp.zeros((nworld, 1, mjm.nM), dtype=wp.float32)
    d.qLD = wp.zeros((nworld, 1, mjm.nM), dtype=wp.float32)
  else:
    d.qM = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=wp.float32)
    d.qLD = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=wp.float32)
  d.act_dot = wp.zeros((nworld, mjm.na), dtype=wp.float32)
  d.act = wp.zeros((nworld, mjm.na), dtype=wp.float32)
  d.qLDiagInv = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.cvel = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector)
  d.cdof_dot = wp.zeros((nworld, mjm.nv), dtype=wp.spatial_vector)
  d.qfrc_bias = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.contact = types.Contact()
  d.contact.dist = wp.zeros((nconmax,), dtype=wp.float32)
  d.contact.pos = wp.zeros((nconmax,), dtype=wp.vec3f)
  d.contact.frame = wp.zeros((nconmax,), dtype=wp.mat33f)
  d.contact.includemargin = wp.zeros((nconmax,), dtype=wp.float32)
  d.contact.friction = wp.zeros((nconmax,), dtype=types.vec5)
  d.contact.solref = wp.zeros((nconmax,), dtype=wp.vec2f)
  d.contact.solreffriction = wp.zeros((nconmax,), dtype=wp.vec2f)
  d.contact.solimp = wp.zeros((nconmax,), dtype=types.vec5)
  d.contact.dim = wp.zeros((nconmax,), dtype=wp.int32)
  d.contact.geom = wp.zeros((nconmax,), dtype=wp.vec2i)
  d.contact.efc_address = wp.zeros((nconmax,), dtype=wp.int32)
  d.contact.worldid = wp.zeros((nconmax,), dtype=wp.int32)
  d.efc = _constraint(mjm.nv, d.nworld, d.njmax)
  d.qfrc_passive = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_spring = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_damper = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_actuator = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_smooth = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_constraint = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_smooth = wp.zeros((nworld, mjm.nv), dtype=wp.float32)

  d.rne_cacc = wp.zeros(shape=(d.nworld, mjm.nbody), dtype=wp.spatial_vector)
  d.rne_cfrc = wp.zeros(shape=(d.nworld, mjm.nbody), dtype=wp.spatial_vector)

  d.xfrc_applied = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector)

  # internal tmp arrays
  d.qfrc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qM_integration = wp.zeros_like(d.qM)
  d.qLD_integration = wp.zeros_like(d.qLD)
  d.qLDiagInv_integration = wp.zeros_like(d.qLDiagInv)
  d.act_vel_integration = wp.zeros_like(d.ctrl)

  # sweep-and-prune broadphase
  d.sap_geom_sort = wp.zeros((nworld, mjm.ngeom), dtype=wp.vec4)
  d.sap_projection_lower = wp.zeros((2 * nworld, mjm.ngeom), dtype=wp.float32)
  d.sap_projection_upper = wp.zeros((nworld, mjm.ngeom), dtype=wp.float32)
  d.sap_sort_index = wp.zeros((2 * nworld, mjm.ngeom), dtype=wp.int32)
  d.sap_range = wp.zeros((nworld, mjm.ngeom), dtype=wp.int32)
  d.sap_cumulative_sum = wp.zeros(nworld * mjm.ngeom, dtype=wp.int32)
  segment_indices_list = [i * mjm.ngeom for i in range(nworld + 1)]
  d.sap_segment_index = wp.array(segment_indices_list, dtype=int)

  # collision driver
  d.collision_pair = wp.empty(nconmax, dtype=wp.vec2i, ndim=1)
  d.collision_worldid = wp.empty(nconmax, dtype=wp.int32, ndim=1)
  d.ncollision = wp.zeros(1, dtype=wp.int32, ndim=1)

  return d


def put_data(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nworld: int = 1,
  nconmax: int = -1,
  njmax: int = -1,
) -> types.Data:
  d = types.Data()
  d.nworld = nworld

  # TODO(team): move to Model?
  if nconmax == -1:
    # TODO(team): heuristic for nconmax
    nconmax = max(512, mjd.ncon * nworld)
  d.nconmax = nconmax
  if njmax == -1:
    # TODO(team): heuristic for njmax
    njmax = max(512, mjd.nefc * nworld)
  d.njmax = njmax

  if nworld * mjd.nefc > njmax:
    raise ValueError("nworld * nefc > njmax")

  d.ncon = wp.array([mjd.ncon * nworld], dtype=wp.int32, ndim=1)
  d.nl = mjd.nl
  d.nefc = wp.array([mjd.nefc * nworld], dtype=wp.int32, ndim=1)
  d.time = mjd.time

  # TODO(erikfrey): would it be better to tile on the gpu?
  def tile(x):
    return np.tile(x, (nworld,) + (1,) * len(x.shape))

  if support.is_sparse(mjm):
    qM = np.expand_dims(mjd.qM, axis=0)
    qLD = np.expand_dims(mjd.qLD, axis=0)
    efc_J = np.zeros((mjd.nefc, mjm.nv))
    mujoco.mju_sparse2dense(
      efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind
    )
  else:
    qM = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, qM, mjd.qM)
    qLD = np.linalg.cholesky(qM)
    efc_J = mjd.efc_J.reshape((mjd.nefc, mjm.nv))

  # TODO(taylorhowell): sparse actuator_moment
  actuator_moment = np.zeros((mjm.nu, mjm.nv))
  mujoco.mju_sparse2dense(
    actuator_moment,
    mjd.actuator_moment,
    mjd.moment_rownnz,
    mjd.moment_rowadr,
    mjd.moment_colind,
  )

  d.qpos = wp.array(tile(mjd.qpos), dtype=wp.float32, ndim=2)
  d.qvel = wp.array(tile(mjd.qvel), dtype=wp.float32, ndim=2)
  d.qacc_warmstart = wp.array(tile(mjd.qacc_warmstart), dtype=wp.float32, ndim=2)
  d.qfrc_applied = wp.array(tile(mjd.qfrc_applied), dtype=wp.float32, ndim=2)
  d.mocap_pos = wp.array(tile(mjd.mocap_pos), dtype=wp.vec3, ndim=2)
  d.mocap_quat = wp.array(tile(mjd.mocap_quat), dtype=wp.quat, ndim=2)
  d.qacc = wp.array(tile(mjd.qacc), dtype=wp.float32, ndim=2)
  d.xanchor = wp.array(tile(mjd.xanchor), dtype=wp.vec3, ndim=2)
  d.xaxis = wp.array(tile(mjd.xaxis), dtype=wp.vec3, ndim=2)
  d.xmat = wp.array(tile(mjd.xmat), dtype=wp.mat33, ndim=2)
  d.xpos = wp.array(tile(mjd.xpos), dtype=wp.vec3, ndim=2)
  d.xquat = wp.array(tile(mjd.xquat), dtype=wp.quat, ndim=2)
  d.xipos = wp.array(tile(mjd.xipos), dtype=wp.vec3, ndim=2)
  d.ximat = wp.array(tile(mjd.ximat), dtype=wp.mat33, ndim=2)
  d.subtree_com = wp.array(tile(mjd.subtree_com), dtype=wp.vec3, ndim=2)
  d.geom_xpos = wp.array(tile(mjd.geom_xpos), dtype=wp.vec3, ndim=2)
  d.geom_xmat = wp.array(tile(mjd.geom_xmat), dtype=wp.mat33, ndim=2)
  d.site_xpos = wp.array(tile(mjd.site_xpos), dtype=wp.vec3, ndim=2)
  d.site_xmat = wp.array(tile(mjd.site_xmat), dtype=wp.mat33, ndim=2)
  d.cinert = wp.array(tile(mjd.cinert), dtype=types.vec10, ndim=2)
  d.cdof = wp.array(tile(mjd.cdof), dtype=wp.spatial_vector, ndim=2)
  d.crb = wp.array(tile(mjd.crb), dtype=types.vec10, ndim=2)
  d.qM = wp.array(tile(qM), dtype=wp.float32, ndim=3)
  d.qLD = wp.array(tile(qLD), dtype=wp.float32, ndim=3)
  d.qLDiagInv = wp.array(tile(mjd.qLDiagInv), dtype=wp.float32, ndim=2)
  d.ctrl = wp.array(tile(mjd.ctrl), dtype=wp.float32, ndim=2)
  d.actuator_velocity = wp.array(tile(mjd.actuator_velocity), dtype=wp.float32, ndim=2)
  d.actuator_force = wp.array(tile(mjd.actuator_force), dtype=wp.float32, ndim=2)
  d.actuator_length = wp.array(tile(mjd.actuator_length), dtype=wp.float32, ndim=2)
  d.actuator_moment = wp.array(tile(actuator_moment), dtype=wp.float32, ndim=3)
  d.cvel = wp.array(tile(mjd.cvel), dtype=wp.spatial_vector, ndim=2)
  d.cdof_dot = wp.array(tile(mjd.cdof_dot), dtype=wp.spatial_vector, ndim=2)
  d.qfrc_bias = wp.array(tile(mjd.qfrc_bias), dtype=wp.float32, ndim=2)
  d.qfrc_passive = wp.array(tile(mjd.qfrc_passive), dtype=wp.float32, ndim=2)
  d.qfrc_spring = wp.array(tile(mjd.qfrc_spring), dtype=wp.float32, ndim=2)
  d.qfrc_damper = wp.array(tile(mjd.qfrc_damper), dtype=wp.float32, ndim=2)
  d.qfrc_actuator = wp.array(tile(mjd.qfrc_actuator), dtype=wp.float32, ndim=2)
  d.qfrc_smooth = wp.array(tile(mjd.qfrc_smooth), dtype=wp.float32, ndim=2)
  d.qfrc_constraint = wp.array(tile(mjd.qfrc_constraint), dtype=wp.float32, ndim=2)
  d.qacc_smooth = wp.array(tile(mjd.qacc_smooth), dtype=wp.float32, ndim=2)
  d.act = wp.array(tile(mjd.act), dtype=wp.float32, ndim=2)
  d.act_dot = wp.array(tile(mjd.act_dot), dtype=wp.float32, ndim=2)

  nefc = mjd.nefc
  efc_worldid = np.zeros(njmax, dtype=int)

  for i in range(nworld):
    efc_worldid[i * nefc : (i + 1) * nefc] = i

  nefc_fill = njmax - nworld * nefc

  efc_J_fill = np.vstack(
    [np.repeat(efc_J, nworld, axis=0), np.zeros((nefc_fill, mjm.nv))]
  )
  efc_D_fill = np.concatenate(
    [np.repeat(mjd.efc_D, nworld, axis=0), np.zeros(nefc_fill)]
  )
  efc_pos_fill = np.concatenate(
    [np.repeat(mjd.efc_pos, nworld, axis=0), np.zeros(nefc_fill)]
  )
  efc_aref_fill = np.concatenate(
    [np.repeat(mjd.efc_aref, nworld, axis=0), np.zeros(nefc_fill)]
  )
  efc_force_fill = np.concatenate(
    [np.repeat(mjd.efc_force, nworld, axis=0), np.zeros(nefc_fill)]
  )
  efc_margin_fill = np.concatenate(
    [np.repeat(mjd.efc_margin, nworld, axis=0), np.zeros(nefc_fill)]
  )

  ncon = mjd.ncon
  con_efc_address = np.zeros(nconmax, dtype=int)
  con_worldid = np.zeros(nconmax, dtype=int)

  for i in range(nworld):
    con_efc_address[i * ncon : (i + 1) * ncon] = mjd.contact.efc_address + i * ncon
    con_worldid[i * ncon : (i + 1) * ncon] = i

  ncon_fill = nconmax - nworld * ncon

  con_dist_fill = np.concatenate(
    [np.repeat(mjd.contact.dist, nworld, axis=0), np.zeros(ncon_fill)]
  )
  con_pos_fill = np.vstack(
    [np.repeat(mjd.contact.pos, nworld, axis=0), np.zeros((ncon_fill, 3))]
  )
  con_frame_fill = np.vstack(
    [np.repeat(mjd.contact.frame, nworld, axis=0), np.zeros((ncon_fill, 9))]
  )
  con_includemargin_fill = np.concatenate(
    [np.repeat(mjd.contact.includemargin, nworld, axis=0), np.zeros(ncon_fill)]
  )
  con_friction_fill = np.vstack(
    [np.repeat(mjd.contact.friction, nworld, axis=0), np.zeros((ncon_fill, 5))]
  )
  con_solref_fill = np.vstack(
    [np.repeat(mjd.contact.solref, nworld, axis=0), np.zeros((ncon_fill, 2))]
  )
  con_solreffriction_fill = np.vstack(
    [np.repeat(mjd.contact.solreffriction, nworld, axis=0), np.zeros((ncon_fill, 2))]
  )
  con_solimp_fill = np.vstack(
    [np.repeat(mjd.contact.solimp, nworld, axis=0), np.zeros((ncon_fill, 5))]
  )
  con_dim_fill = np.concatenate(
    [np.repeat(mjd.contact.dim, nworld, axis=0), np.zeros(ncon_fill)]
  )
  con_geom_fill = np.vstack(
    [np.repeat(mjd.contact.geom, nworld, axis=0), np.zeros((ncon_fill, 2))]
  )

  d.contact.dist = wp.array(con_dist_fill, dtype=wp.float32, ndim=1)
  d.contact.pos = wp.array(con_pos_fill, dtype=wp.vec3f, ndim=1)
  d.contact.frame = wp.array(con_frame_fill, dtype=wp.mat33f, ndim=1)
  d.contact.includemargin = wp.array(con_includemargin_fill, dtype=wp.float32, ndim=1)
  d.contact.friction = wp.array(con_friction_fill, dtype=types.vec5, ndim=1)
  d.contact.solref = wp.array(con_solref_fill, dtype=wp.vec2f, ndim=1)
  d.contact.solreffriction = wp.array(con_solreffriction_fill, dtype=wp.vec2f, ndim=1)
  d.contact.solimp = wp.array(con_solimp_fill, dtype=types.vec5, ndim=1)
  d.contact.dim = wp.array(con_dim_fill, dtype=wp.int32, ndim=1)
  d.contact.geom = wp.array(con_geom_fill, dtype=wp.vec2i, ndim=1)
  d.contact.efc_address = wp.array(con_efc_address, dtype=wp.int32, ndim=1)
  d.contact.worldid = wp.array(con_worldid, dtype=wp.int32, ndim=1)

  d.rne_cacc = wp.zeros(shape=(d.nworld, mjm.nbody), dtype=wp.spatial_vector)
  d.rne_cfrc = wp.zeros(shape=(d.nworld, mjm.nbody), dtype=wp.spatial_vector)

  d.efc = _constraint(mjm.nv, d.nworld, d.njmax)
  d.efc.J = wp.array(efc_J_fill, dtype=wp.float32, ndim=2)
  d.efc.D = wp.array(efc_D_fill, dtype=wp.float32, ndim=1)
  d.efc.pos = wp.array(efc_pos_fill, dtype=wp.float32, ndim=1)
  d.efc.aref = wp.array(efc_aref_fill, dtype=wp.float32, ndim=1)
  d.efc.force = wp.array(efc_force_fill, dtype=wp.float32, ndim=1)
  d.efc.margin = wp.array(efc_margin_fill, dtype=wp.float32, ndim=1)
  d.efc.worldid = wp.from_numpy(efc_worldid, dtype=wp.int32)

  d.xfrc_applied = wp.array(tile(mjd.xfrc_applied), dtype=wp.spatial_vector, ndim=2)

  # internal tmp arrays
  d.qfrc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qM_integration = wp.zeros_like(d.qM)
  d.qLD_integration = wp.zeros_like(d.qLD)
  d.qLDiagInv_integration = wp.zeros_like(d.qLDiagInv)
  d.act_vel_integration = wp.zeros_like(d.ctrl)

  # broadphase sweep and prune
  d.sap_geom_sort = wp.zeros((nworld, mjm.ngeom), dtype=wp.vec4)
  d.sap_projection_lower = wp.zeros((2 * nworld, mjm.ngeom), dtype=wp.float32)
  d.sap_projection_upper = wp.zeros((nworld, mjm.ngeom), dtype=wp.float32)
  d.sap_sort_index = wp.zeros((2 * nworld, mjm.ngeom), dtype=wp.int32)
  d.sap_range = wp.zeros((nworld, mjm.ngeom), dtype=wp.int32)
  d.sap_cumulative_sum = wp.zeros(nworld * mjm.ngeom, dtype=wp.int32)
  segment_indices_list = [i * mjm.ngeom for i in range(nworld + 1)]
  d.sap_segment_index = wp.array(segment_indices_list, dtype=int)

  # collision driver
  d.collision_pair = wp.empty(nconmax, dtype=wp.vec2i, ndim=1)
  d.collision_worldid = wp.empty(nconmax, dtype=wp.int32, ndim=1)
  d.ncollision = wp.zeros(1, dtype=wp.int32, ndim=1)

  return d


def get_data_into(
  result: mujoco.MjData,
  mjm: mujoco.MjModel,
  d: types.Data,
):
  """Gets Data from a device into an existing mujoco.MjData."""
  if d.nworld > 1:
    raise NotImplementedError("only nworld == 1 supported for now")

  ncon = d.ncon.numpy()[0]
  nefc = d.nefc.numpy()[0]

  if ncon != result.ncon or nefc != result.nefc:
    mujoco._functions._realloc_con_efc(result, ncon=ncon, nefc=nefc)

  result.time = d.time

  result.qpos[:] = d.qpos.numpy()[0]
  result.qvel[:] = d.qvel.numpy()[0]
  result.qacc_warmstart = d.qacc_warmstart.numpy()[0]
  result.qfrc_applied = d.qfrc_applied.numpy()[0]
  result.mocap_pos = d.mocap_pos.numpy()[0]
  result.mocap_quat = d.mocap_quat.numpy()[0]
  result.qacc = d.qacc.numpy()[0]
  result.xanchor = d.xanchor.numpy()[0]
  result.xaxis = d.xaxis.numpy()[0]
  result.xmat = d.xmat.numpy().reshape((-1, 9))
  result.xpos = d.xpos.numpy()[0]
  result.xquat = d.xquat.numpy()[0]
  result.xipos = d.xipos.numpy()[0]
  result.ximat = d.ximat.numpy().reshape((-1, 9))
  result.subtree_com = d.subtree_com.numpy()[0]
  result.geom_xpos = d.geom_xpos.numpy()[0]
  result.geom_xmat = d.geom_xmat.numpy().reshape((-1, 9))
  result.site_xpos = d.site_xpos.numpy()[0]
  result.site_xmat = d.site_xmat.numpy().reshape((-1, 9))
  result.cinert = d.cinert.numpy()[0]
  result.cdof = d.cdof.numpy()[0]
  result.crb = d.crb.numpy()[0]
  result.qLDiagInv = d.qLDiagInv.numpy()[0]
  result.ctrl = d.ctrl.numpy()[0]
  result.actuator_velocity = d.actuator_velocity.numpy()[0]
  result.actuator_force = d.actuator_force.numpy()[0]
  result.actuator_length = d.actuator_length.numpy()[0]
  mujoco.mju_dense2sparse(
    result.actuator_moment,
    d.actuator_moment.numpy()[0],
    result.moment_rownnz,
    result.moment_rowadr,
    result.moment_colind,
  )
  result.cvel = d.cvel.numpy()[0]
  result.cdof_dot = d.cdof_dot.numpy()[0]
  result.qfrc_bias = d.qfrc_bias.numpy()[0]
  result.qfrc_passive = d.qfrc_passive.numpy()[0]
  result.qfrc_spring = d.qfrc_spring.numpy()[0]
  result.qfrc_damper = d.qfrc_damper.numpy()[0]
  result.qfrc_actuator = d.qfrc_actuator.numpy()[0]
  result.qfrc_smooth = d.qfrc_smooth.numpy()[0]
  result.qfrc_constraint = d.qfrc_constraint.numpy()[0]
  result.qacc_smooth = d.qacc_smooth.numpy()[0]
  result.act = d.act.numpy()[0]
  result.act_dot = d.act_dot.numpy()[0]

  result.contact.dist[:] = d.contact.dist.numpy()[:ncon]
  result.contact.pos[:] = d.contact.pos.numpy()[:ncon]
  result.contact.frame[:] = d.contact.frame.numpy()[:ncon].reshape((-1, 9))
  result.contact.includemargin[:] = d.contact.includemargin.numpy()[:ncon]
  result.contact.friction[:] = d.contact.friction.numpy()[:ncon]
  result.contact.solref[:] = d.contact.solref.numpy()[:ncon]
  result.contact.solreffriction[:] = d.contact.solreffriction.numpy()[:ncon]
  result.contact.solimp[:] = d.contact.solimp.numpy()[:ncon]
  result.contact.dim[:] = d.contact.dim.numpy()[:ncon]
  result.contact.efc_address[:] = d.contact.efc_address.numpy()[:ncon]

  if support.is_sparse(mjm):
    result.qM[:] = d.qM.numpy()[0, 0]
    result.qLD[:] = d.qLD.numpy()[0, 0]
    # TODO(team): set efc_J after fix to _realloc_con_efc lands
    # efc_J = d.efc_J.numpy()[0, :nefc]
    # mujoco.mju_dense2sparse(
    #   result.efc_J, efc_J, result.efc_J_rownnz, result.efc_J_rowadr, result.efc_J_colind
    # )
  else:
    qM = d.qM.numpy()
    qLD = d.qLD.numpy()
    adr = 0
    for i in range(mjm.nv):
      j = i
      while j >= 0:
        result.qM[adr] = qM[0, i, j]
        result.qLD[adr] = qLD[0, i, j]
        j = mjm.dof_parentid[j]
        adr += 1
    # TODO(team): set efc_J after fix to _realloc_con_efc lands
    # if nefc > 0:
    #   result.efc_J[:nefc * mjm.nv] = d.efc_J.numpy()[:nefc].flatten()
  result.efc_D[:] = d.efc.D.numpy()[:nefc]
  result.efc_pos[:] = d.efc.pos.numpy()[:nefc]
  result.efc_aref[:] = d.efc.aref.numpy()[:nefc]
  result.efc_force[:] = d.efc.force.numpy()[:nefc]
  result.efc_margin[:] = d.efc.margin.numpy()[:nefc]

  # TODO: other efc_ fields, anything else missing
