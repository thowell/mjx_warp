import warp as wp
from . import passive
from . import smooth

from .types import Model
from .types import Data


def fwd_position(m: Model, d: Data):
  """Position-dependent computations."""

  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  # TODO(team): smooth.camlight
  # TODO(team): smooth.tendon
  smooth.crb(m, d)
  smooth.factor_m(m, d)
  # TODO(team): collision_driver.collision
  # TODO(team): constraint.make_constraint
  # TODO(team): smooth.transmission


def fwd_velocity(m: Model, d: Data):
  """Velocity-dependent computations."""

  # TODO(team): tile operations?
  @wp.kernel
  def _actuator_velocity(d: Data):
    worldid, actid, dofid = wp.tid()
    moment = d.actuator_moment[worldid, actid]
    qvel = d.qvel[worldid]
    wp.atomic_add(d.actuator_velocity[worldid], actid, moment[dofid] * qvel[dofid])

  wp.launch(_actuator_velocity, dim=(d.nworld, m.nu, m.nv), inputs=[d])

  smooth.com_vel(m, d)
  passive.passive(m, d)
  smooth.rne(m, d)


def fwd_acceleration(m: Model, d: Data):
  """Add up all non-constraint forces, compute qacc_smooth."""

  qfrc_applied = d.qfrc_applied
  # TODO(team) += support.xfrc_accumulate(m, d)

  @wp.kernel
  def _qfrc_smooth(d: Data, qfrc_applied: wp.array(ndim=2, dtype=wp.float32)):
    worldid, dofid = wp.tid()
    d.qfrc_smooth[worldid, dofid] = (
      d.qfrc_passive[worldid, dofid]
      - d.qfrc_bias[worldid, dofid]
      + d.qfrc_actuator[worldid, dofid]
      + qfrc_applied[worldid, dofid]
    )

  wp.launch(_qfrc_smooth, dim=(d.nworld, m.nv), inputs=[d, qfrc_applied])

  smooth.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)


def forward(m: Model, d: Data):
  """Forward dynamics."""

  fwd_position(m, d)
  # TODO(team): sensor.sensor_pos
  # TODO(taylorhowell): fwd_velocity
  # TODO(team): sensor.sensor_vel
  # TODO(team): fwd_actuation
  fwd_acceleration(m, d)
  # TODO(team): sensor.sensor_acc

  # if nefc == 0
  wp.copy(d.qacc, d.qacc_smooth)

  # TODO(team): solver.solve
