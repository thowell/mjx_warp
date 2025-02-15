import warp as wp
from . import smooth
from . import types


def fwd_acceleration(m: types.Model, d: types.Data):
  """Add up all non-constraint forces, compute qacc_smooth."""

  qfrc_applied = d.qfrc_applied
  # TODO(team) += support.xfrc_accumulate(m, d)

  @wp.kernel
  def _qfrc_smooth(d: types.Data, qfrc_applied: wp.array(ndim=2, dtype=wp.float32)):
    worldid, dofid = wp.tid()
    d.qfrc_smooth[worldid, dofid] = d.qfrc_passive[worldid, dofid] - d.qfrc_bias[worldid, dofid] + d.qfrc_actuator[worldid, dofid] + qfrc_applied[worldid, dofid]

  wp.launch(_qfrc_smooth, dim=(d.nworld, m.nv), inputs=[d, qfrc_applied])

  smooth.solve_m(m, d, d.qfrc_smooth, d.qacc_smooth)
