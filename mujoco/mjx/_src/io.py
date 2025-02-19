import warp as wp
import mujoco
import numpy as np

from . import support
from . import types


def put_model(mjm: mujoco.MjModel) -> types.Model:
  m = types.Model()
  m.nq = mjm.nq
  m.nv = mjm.nv
  m.nbody = mjm.nbody
  m.njnt = mjm.njnt
  m.ngeom = mjm.ngeom
  m.nsite = mjm.nsite
  m.nmocap = mjm.nmocap
  m.nM = mjm.nM
  m.qpos0 = wp.array(mjm.qpos0, dtype=wp.float32, ndim=1)
  m.qpos_spring = wp.array(mjm.qpos_spring, dtype=wp.float32, ndim=1)

  # body_tree is BFS ordering of body ids
  body_tree, body_depth = {}, np.zeros(mjm.nbody, dtype=int) - 1
  for i in range(mjm.nbody):
    body_depth[i] = body_depth[mjm.body_parentid[i]] + 1
    body_tree.setdefault(body_depth[i], []).append(i)
  # body_leveladr, body_levelsize specify the bounds of level ranges in body_level
  body_levelsize = np.array([len(body_tree[i]) for i in range(len(body_tree))])
  body_leveladr = np.cumsum(np.insert(body_levelsize, 0, 0))[:-1]
  body_tree = sum([body_tree[i] for i in range(len(body_tree))], [])

  qLD_sparse_updates = np.empty(shape=(0, 3), dtype=int)
  qld_dense_tilesize = np.empty(shape=(0,), dtype=int)
  qld_dense_tileid = np.empty(shape=(0,), dtype=int)
  if support.is_sparse(mjm):
    # track qLD updates for factor_m
    qLD_updates, dof_depth = {}, np.zeros(mjm.nv, dtype=int) - 1
    for k in range(mjm.nv):
      dof_depth[k] = dof_depth[mjm.dof_parentid[k]] + 1
      i = mjm.dof_parentid[k]
      Madr_ki = mjm.dof_Madr[k] + 1
      while i > -1:
        qLD_updates.setdefault(dof_depth[i], []).append((i, k, Madr_ki))
        i = mjm.dof_parentid[i]
        Madr_ki += 1

    # qLD_leveladr, qLD_levelsize specify the bounds of level ranges in qLD updates
    qLD_levelsize = np.array([len(qLD_updates[i]) for i in range(len(qLD_updates))])
    qLD_leveladr = np.cumsum(np.insert(qLD_levelsize, 0, 0))[:-1]
    qLD_sparse_updates = np.array(sum([qLD_updates[i] for i in range(len(qLD_updates))], []))
  else:
    # track tile sizes for dense cholesky
    tile_corners = [i for i in range(mjm.nv) if mjm.dof_parentid[i] == -1]
    tiles = {}
    for i in range(len(tile_corners)):
      tile_beg = tile_corners[i]
      tile_end = mjm.nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
      tiles.setdefault(tile_end - tile_beg, []).append(tile_beg)
    # qLD_leveladr, qLD_levelsize specify the bounds of level ranges in cholesky tiles
    qLD_levelsize = np.array([len(tiles[sz]) for sz in sorted(tiles.keys())])
    qLD_leveladr = np.cumsum(np.insert(qLD_levelsize, 0, 0))[:-1]
    qld_dense_tilesize = np.array(sorted(tiles.keys()))
    qld_dense_tileid = np.array(sum([tiles[sz] for sz in sorted(tiles.keys())], []))

  m.body_leveladr = wp.array(body_leveladr, dtype=wp.int32, ndim=1, device="cpu")
  m.body_levelsize = wp.array(body_levelsize, dtype=wp.int32, ndim=1, device="cpu")
  m.body_tree = wp.array(body_tree, dtype=wp.int32, ndim=1)
  m.qLD_leveladr = wp.array(qLD_leveladr, dtype=wp.int32, ndim=1, device="cpu")
  m.qLD_levelsize = wp.array(qLD_levelsize, dtype=wp.int32, ndim=1, device="cpu")
  m.qLD_sparse_updates = wp.array(qLD_sparse_updates, dtype=wp.vec3i, ndim=1)
  m.qLD_dense_tilesize = wp.array(qld_dense_tilesize, dtype=wp.int32, ndim=1, device="cpu")
  m.qLD_dense_tileid = wp.array(qld_dense_tileid, dtype=wp.int32, ndim=1)
  m.body_dofadr = wp.array(mjm.body_dofadr, dtype=wp.int32, ndim=1)
  m.body_dofnum = wp.array(mjm.body_dofnum, dtype=wp.int32, ndim=1)
  m.body_jntadr = wp.array(mjm.body_jntadr, dtype=wp.int32, ndim=1)
  m.body_jntnum = wp.array(mjm.body_jntnum, dtype=wp.int32, ndim=1)
  m.body_parentid = wp.array(mjm.body_parentid, dtype=wp.int32, ndim=1)
  m.body_mocapid = wp.array(mjm.body_mocapid, dtype=wp.int32, ndim=1)
  m.body_pos = wp.array(mjm.body_pos, dtype=wp.vec3, ndim=1)
  m.body_quat = wp.array(mjm.body_quat, dtype=wp.quat, ndim=1)
  m.body_ipos = wp.array(mjm.body_ipos, dtype=wp.vec3, ndim=1)
  m.body_iquat = wp.array(mjm.body_iquat, dtype=wp.quat, ndim=1)
  m.body_rootid = wp.array(mjm.body_rootid, dtype=wp.int32, ndim=1)
  m.body_inertia = wp.array(mjm.body_inertia, dtype=wp.vec3, ndim=1)
  m.body_mass = wp.array(mjm.body_mass, dtype=wp.float32, ndim=1)
  m.jnt_bodyid = wp.array(mjm.jnt_bodyid, dtype=wp.int32, ndim=1)
  m.jnt_type = wp.array(mjm.jnt_type, dtype=wp.int32, ndim=1)
  m.jnt_qposadr = wp.array(mjm.jnt_qposadr, dtype=wp.int32, ndim=1)
  m.jnt_dofadr = wp.array(mjm.jnt_dofadr, dtype=wp.int32, ndim=1)
  m.jnt_axis = wp.array(mjm.jnt_axis, dtype=wp.vec3, ndim=1)
  m.jnt_pos = wp.array(mjm.jnt_pos, dtype=wp.vec3, ndim=1)
  m.jnt_stiffness = wp.array(mjm.jnt_stiffness, dtype=wp.float32, ndim=1)
  m.geom_pos = wp.array(mjm.geom_pos, dtype=wp.vec3, ndim=1)
  m.geom_quat = wp.array(mjm.geom_quat, dtype=wp.quat, ndim=1)
  m.site_pos = wp.array(mjm.site_pos, dtype=wp.vec3, ndim=1)
  m.site_quat = wp.array(mjm.site_quat, dtype=wp.quat, ndim=1)
  m.dof_bodyid = wp.array(mjm.dof_bodyid, dtype=wp.int32, ndim=1)
  m.dof_jntid = wp.array(mjm.dof_jntid, dtype=wp.int32, ndim=1)
  m.dof_parentid = wp.array(mjm.dof_parentid, dtype=wp.int32, ndim=1)
  m.dof_Madr = wp.array(mjm.dof_Madr, dtype=wp.int32, ndim=1)
  m.dof_armature = wp.array(mjm.dof_armature, dtype=wp.float32, ndim=1)
  m.dof_damping = wp.array(mjm.dof_damping, dtype=wp.float32, ndim=1)
  m.opt.gravity = wp.vec3(mjm.opt.gravity)
  m.opt.tolerance = mjm.opt.tolerance
  m.opt.ls_tolerance = mjm.opt.ls_tolerance
  m.opt.cone = mjm.opt.cone
  m.opt.solver = mjm.opt.solver
  m.opt.iterations = mjm.opt.iterations
  m.opt.ls_iterations = mjm.opt.ls_iterations
  m.opt.is_sparse = support.is_sparse(mjm)
  m.stat.meaninertia = mjm.stat.meaninertia

  return m


def make_data(mjm: mujoco.MjModel, nworld: int = 1, nefc_maxbatch: int = 512) -> types.Data:
  d = types.Data()
  d.nworld = nworld
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
  d.crb = wp.zeros((nworld, mjm.nbody), dtype=types.vec10)
  if support.is_sparse(mjm):
    d.qM = wp.zeros((nworld, 1, mjm.nM), dtype=wp.float32)
    d.qLD = wp.zeros((nworld, 1, mjm.nM), dtype=wp.float32)
  else:
    d.qM = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=wp.float32)
    d.qLD = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=wp.float32)
  d.qLDiagInv = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.cvel = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector)
  d.cdof_dot = wp.zeros((nworld, mjm.nv), dtype=wp.spatial_vector)
  d.qfrc_bias = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_passive = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_spring = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_damper = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_actuator = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_smooth = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_smooth = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_constraint = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.nefc_active = 0
  d.nefc_maxbatch = nefc_maxbatch
  d.efc_J = wp.zeros((nefc_maxbatch, mjm.nv), dtype=wp.float32)
  d.efc_D = wp.zeros((nefc_maxbatch), dtype=wp.float32)
  d.efc_aref = wp.zeros((nefc_maxbatch), dtype=wp.float32)
  d.efc_force = wp.zeros((nefc_maxbatch), dtype=wp.float32)
  d.efc_worldid = wp.zeros((nefc_maxbatch), dtype=wp.int32)
  d.world_efcadr = wp.zeros((nworld), dtype=wp.int32)
  d.world_efcsize = wp.zeros((nworld), dtype=wp.int32)

  return d


def put_data(mjm: mujoco.MjModel, mjd: mujoco.MjData, nworld: int = 1, nefc_maxbatch: int = 512) -> types.Data:
  d = types.Data()

  if nworld * mjd.nefc > nefc_maxbatch:
    raise ValueError("nworld * nefc > nefc_maxbatch")
  
  d.nworld = nworld
  # TODO(erikfrey): would it be better to tile on the gpu?
  def tile(x):
    return np.tile(x, (nworld,) + (1,) * len(x.shape))

  if support.is_sparse(mjm):
    qM = np.expand_dims(mjd.qM, axis=0)
    qLD = np.expand_dims(mjd.qLD, axis=0)
  else:
    qM = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, qM, mjd.qM)
    qLD = np.linalg.cholesky(qM, upper=True)

  efc_J = mjd.efc_J.reshape((mjd.nefc, mjm.nv))

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
  d.cvel = wp.array(tile(mjd.cvel), dtype=wp.spatial_vector, ndim=2)
  d.cdof_dot = wp.array(tile(mjd.cdof_dot), dtype=wp.spatial_vector, ndim=2)
  d.qfrc_bias = wp.array(tile(mjd.qfrc_bias), dtype=wp.float32, ndim=2)
  d.qfrc_passive = wp.array(tile(mjd.qfrc_passive), dtype=wp.float32, ndim=2)
  d.qfrc_spring = wp.array(tile(mjd.qfrc_spring), dtype=wp.float32, ndim=2)
  d.qfrc_damper = wp.array(tile(mjd.qfrc_damper), dtype=wp.float32, ndim=2)
  d.qfrc_actuator = wp.array(tile(mjd.qfrc_actuator), dtype=wp.float32, ndim=2)
  d.qfrc_smooth = wp.array(tile(mjd.qfrc_smooth), dtype=wp.float32, ndim=2)
  d.qacc_smooth = wp.array(tile(mjd.qacc_smooth), dtype=wp.float32, ndim=2)
  d.qfrc_constraint = wp.array(tile(mjd.qfrc_constraint), dtype=wp.float32, ndim=2)

  nefc = mjd.nefc
  d.nefc_active = nworld * nefc
  d.nefc_maxbatch = nefc_maxbatch
  efc_worldid = np.zeros(nefc_maxbatch, dtype=int)
  world_efcadr = np.zeros(nworld, dtype=int)
  world_efcsize = np.zeros(nworld, dtype=int)


  for i in range(nworld):
    efc_worldid[i * nefc: (i + 1) * nefc] = i
    if i > 0:
      world_efcadr[i] = world_efcadr[i - 1] + nefc
    else:
      world_efcadr[i] = 0
    world_efcsize[i] = nefc

  nefc_fill = nefc_maxbatch - nworld * nefc

  efc_J_fill = np.vstack([np.repeat(efc_J, nworld, axis=0), np.zeros((nefc_fill, mjm.nv))])
  efc_D_fill = np.concatenate([np.repeat(mjd.efc_D, nworld, axis=0), np.zeros(nefc_fill)])
  efc_aref_fill = np.concatenate([np.repeat(mjd.efc_aref, nworld, axis=0), np.zeros(nefc_fill)])
  efc_force_fill = np.concatenate([np.repeat(mjd.efc_force, nworld, axis=0), np.zeros(nefc_fill)])

  d.efc_J = wp.array(efc_J_fill, dtype=wp.float32, ndim=2)
  d.efc_D = wp.array(efc_D_fill, dtype=wp.float32, ndim=1)
  d.efc_aref = wp.array(efc_aref_fill, dtype=wp.float32, ndim=1)
  d.efc_force = wp.array(efc_force_fill, dtype=wp.float32, ndim=1)
  d.efc_worldid = wp.from_numpy(efc_worldid, dtype=wp.int32)
  d.world_efcadr = wp.from_numpy(world_efcadr, dtype=wp.int32)
  d.world_efcsize = wp.from_numpy(world_efcsize, dtype=wp.int32)

  return d
