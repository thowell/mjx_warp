import warp as wp
import mujoco
import numpy as np

from . import types


def put_model(m: mujoco.MjModel) -> types.Model:
  mx = types.Model()
  mx.nq = m.nq
  mx.nv = m.nv
  mx.nbody = m.nbody
  mx.njnt = m.njnt
  mx.ngeom = m.ngeom
  mx.nsite = m.nsite
  mx.nmocap = m.nmocap
  mx.qpos0 = wp.array(m.qpos0, dtype=wp.float32, ndim=2)

  # body_bfs is BFS ordering of body ids
  # level_beg, level_end specify the bounds of level ranges in body_range
  level_beg, level_end, body_bfs = [], [], []
  parents = {0}
  while len(body_bfs) < m.nbody - 1:
    children = [i for i, p in enumerate(m.body_parentid) if p in parents and i != 0]
    if not children:
      raise ValueError("invalid tree layout")
    level_beg.append(len(body_bfs))
    body_bfs.extend(children)
    level_end.append(len(body_bfs))
    parents = set(children)

  mx.nlevel = len(level_beg)
  mx.level_beg = wp.array(level_beg, dtype=wp.int32, ndim=1)
  mx.level_beg_cpu = wp.array(level_beg, dtype=wp.int32, ndim=1, device='cpu')
  mx.level_end = wp.array(level_end, dtype=wp.int32, ndim=1)
  mx.level_end_cpu = wp.array(level_end, dtype=wp.int32, ndim=1, device='cpu')
  mx.body_bfs = wp.array(body_bfs, dtype=wp.int32, ndim=1)
  mx.body_jntadr = wp.array(m.body_jntadr, dtype=wp.int32, ndim=1)
  mx.body_jntnum = wp.array(m.body_jntnum, dtype=wp.int32, ndim=1)
  mx.body_parentid = wp.array(m.body_parentid, dtype=wp.int32, ndim=1)
  mx.body_mocapid = wp.array(m.body_mocapid, dtype=wp.int32, ndim=1)
  mx.body_pos = wp.array(m.body_pos, dtype=wp.vec3, ndim=1)
  mx.body_quat = wp.array(m.body_quat, dtype=wp.quat, ndim=1)
  mx.body_ipos = wp.array(m.body_ipos, dtype=wp.vec3, ndim=1)
  mx.body_iquat = wp.array(m.body_iquat, dtype=wp.quat, ndim=1)
  mx.jnt_type = wp.array(m.jnt_type, dtype=wp.int32, ndim=1)
  mx.jnt_qposadr = wp.array(m.jnt_qposadr, dtype=wp.int32, ndim=1)
  mx.jnt_axis = wp.array(m.jnt_axis, dtype=wp.vec3, ndim=1)
  mx.jnt_pos = wp.array(m.jnt_pos, dtype=wp.vec3, ndim=1)
  mx.geom_pos = wp.array(m.geom_pos, dtype=wp.vec3, ndim=1)
  mx.geom_quat = wp.array(m.geom_quat, dtype=wp.quat, ndim=1)
  mx.site_pos = wp.array(m.site_pos, dtype=wp.vec3, ndim=1)
  mx.site_quat = wp.array(m.site_quat, dtype=wp.quat, ndim=1)

  return mx

def make_data(m: mujoco.MjModel, nworld: int = 1) -> types.Data:
  d = types.Data()
  d.nworld = nworld

  qpos0 = np.tile(m.qpos0, (nworld, 1))
  d.qpos = wp.array(qpos0, dtype=wp.float32, ndim=2)
  d.mocap_pos = wp.zeros((nworld, m.nmocap), dtype=wp.vec3)
  d.mocap_quat = wp.zeros((nworld, m.nmocap), dtype=wp.quat)
  d.xanchor = wp.zeros((nworld, m.njnt), dtype=wp.vec3)
  d.xaxis = wp.zeros((nworld, m.njnt), dtype=wp.vec3)
  d.xmat = wp.zeros((nworld, m.nbody), dtype=wp.mat33)
  d.xpos = wp.zeros((nworld, m.nbody), dtype=wp.vec3)
  d.xquat = wp.zeros((nworld, m.nbody), dtype=wp.quat)
  d.xipos = wp.zeros((nworld, m.nbody), dtype=wp.vec3)
  d.ximat = wp.zeros((nworld, m.nbody), dtype=wp.mat33)
  d.geom_xpos = wp.zeros((nworld, m.ngeom), dtype=wp.vec3)
  d.geom_xmat = wp.zeros((nworld, m.ngeom), dtype=wp.mat33)
  d.site_xpos = wp.zeros((nworld, m.nsite), dtype=wp.vec3)
  d.site_xmat = wp.zeros((nworld, m.nsite), dtype=wp.mat33)

  return d
