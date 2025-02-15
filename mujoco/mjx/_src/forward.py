import warp as wp
from . import passive
from . import smooth
from . import types


def fwd_velocity(m: types.Model, d: types.Data):
  """Velocity-dependent computations."""

  # TODO(team): tile operations?
  @wp.kernel
  def _actuator_velocity(d: types.Data):
    worldid, actid, dofid = wp.tid()
    moment = d.actuator_moment[worldid, actid]
    qvel = d.qvel[worldid]
    wp.atomic_add(d.actuator_velocity[worldid],
                  actid, moment[dofid] * qvel[dofid])

  wp.launch(_actuator_velocity, dim=(d.nworld, m.nu, m.nv), inputs=[d])

  # TODO(taylorhowell): smooth.com_vel
  passive.passive(m, d)
  smooth.rne(m, d)
