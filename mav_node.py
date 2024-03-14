#!/usr/bin/env python3
import sys
import math
import time
import argparse
from enum import Enum

import pandas
import numpy as np
from numpy import deg2rad
from numpy import rad2deg
from numpy import cos
from numpy import sin
import matplotlib.pylab as plt

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.msg import Thrust
from mavros_msgs.msg import ParamValue
from mavros_msgs.srv import ParamSet
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import CommandBoolRequest
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import SetModeRequest


def clip_value(x, vmin, vmax):
  """ Clip """
  x_tmp = x
  x_tmp = vmax if (x_tmp > vmax) else x_tmp
  x_tmp = vmin if (x_tmp < vmin) else x_tmp
  return x_tmp


def quat_norm(q):
  """ Returns norm of a quaternion """
  qw, qx, qy, qz = q
  return sqrt(qw**2 + qx**2 + qy**2 + qz**2)


def quat_normalize(q):
  """ Normalize quaternion """
  n = quat_norm(q)
  qw, qx, qy, qz = q
  return np.array([qw / n, qx / n, qy / n, qz / n])


def quat_conj(q):
  """ Return conjugate quaternion """
  qw, qx, qy, qz = q
  q_conj = np.array([qw, -qx, -qy, -qz])
  return q_conj


def quat_inv(q):
  """ Invert quaternion """
  return quat_conj(q)


def quat_left(q):
  """ Quaternion left product matrix """
  qw, qx, qy, qz = q
  row0 = [qw, -qx, -qy, -qz]
  row1 = [qx, qw, -qz, qy]
  row2 = [qy, qz, qw, -qx]
  row3 = [qz, -qy, qx, qw]
  return np.array([row0, row1, row2, row3])


def quat_right(q):
  """ Quaternion right product matrix """
  qw, qx, qy, qz = q
  row0 = [qw, -qx, -qy, -qz]
  row1 = [qx, qw, qz, -qy]
  row2 = [qy, -qz, qw, qx]
  row3 = [qz, qy, -qx, qw]
  return np.array([row0, row1, row2, row3])


def quat_lmul(p, q):
  """ Quaternion left multiply """
  assert len(p) == 4
  assert len(q) == 4
  lprod = quat_left(p)
  return lprod @ q


def quat_rmul(p, q):
  """ Quaternion right multiply """
  assert len(p) == 4
  assert len(q) == 4
  rprod = quat_right(q)
  return rprod @ p


def quat_mul(p, q):
  """ Quaternion multiply p * q """
  return quat_lmul(p, q)


def quat2euler(q):
  """
  Convert quaternion to euler angles (yaw, pitch, roll).

  Source:
  Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
  Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
  Princeton University Press, 1999. Print.
  [Page 168, "Quaternion to Euler Angles"]
  """
  qw, qx, qy, qz = q

  m11 = (2 * qw**2) + (2 * qx**2) - 1
  m12 = 2 * (qx * qy + qw * qz)
  m13 = 2 * qx * qz - 2 * qw * qy
  m23 = 2 * qy * qz + 2 * qw * qx
  m33 = (2 * qw**2) + (2 * qz**2) - 1

  psi = math.atan2(m12, m11)
  theta = math.asin(-m13)
  phi = math.atan2(m23, m33)

  ypr = np.array([psi, theta, phi])
  return ypr


def euler2quat(yaw, pitch, roll):
  """
  Convert yaw, pitch, roll in radians to a quaternion.

  Source:
  Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
  Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
  Princeton University Press, 1999. Print.
  [Page 166-167, "Euler Angles to Quaternion"]
  """
  psi = yaw  # Yaw
  theta = pitch  # Pitch
  phi = roll  # Roll

  c_phi = cos(phi / 2.0)
  c_theta = cos(theta / 2.0)
  c_psi = cos(psi / 2.0)
  s_phi = sin(phi / 2.0)
  s_theta = sin(theta / 2.0)
  s_psi = sin(psi / 2.0)

  qw = c_psi * c_theta * c_phi + s_psi * s_theta * s_phi
  qx = c_psi * c_theta * s_phi - s_psi * s_theta * c_phi
  qy = c_psi * s_theta * c_phi + s_psi * c_theta * s_phi
  qz = s_psi * c_theta * c_phi - c_psi * s_theta * s_phi

  mag = sqrt(qw**2 + qx**2 + qy**2 + qz**2)
  return np.array([qw / mag, qx / mag, qy / mag, qz / mag])


def euler321(yaw, pitch, roll):
  """
  Convert yaw, pitch, roll in radians to a 3x3 rotation matrix.

  Source:
  Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
  Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
  Princeton University Press, 1999. Print.
  [Page 85-86, "The Aerospace Sequence"]
  """
  psi = yaw
  theta = pitch
  phi = roll

  cpsi = cos(psi)
  spsi = sin(psi)
  ctheta = cos(theta)
  stheta = sin(theta)
  cphi = cos(phi)
  sphi = sin(phi)

  C11 = cpsi * ctheta
  C21 = spsi * ctheta
  C31 = -stheta

  C12 = cpsi * stheta * sphi - spsi * cphi
  C22 = spsi * stheta * sphi + cpsi * cphi
  C32 = ctheta * sphi

  C13 = cpsi * stheta * cphi + spsi * sphi
  C23 = spsi * stheta * cphi - cpsi * sphi
  C33 = ctheta * cphi

  return np.array([[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]])


def rot2quat(C):
  """
  Convert 3x3 rotation matrix to quaternion.
  """
  assert C.shape == (3, 3)

  m00 = C[0, 0]
  m01 = C[0, 1]
  m02 = C[0, 2]

  m10 = C[1, 0]
  m11 = C[1, 1]
  m12 = C[1, 2]

  m20 = C[2, 0]
  m21 = C[2, 1]
  m22 = C[2, 2]

  tr = m00 + m11 + m22

  if tr > 0:
    S = sqrt(tr + 1.0) * 2.0
    # S=4*qw
    qw = 0.25 * S
    qx = (m21 - m12) / S
    qy = (m02 - m20) / S
    qz = (m10 - m01) / S
  elif ((m00 > m11) and (m00 > m22)):
    S = sqrt(1.0 + m00 - m11 - m22) * 2.0
    # S=4*qx
    qw = (m21 - m12) / S
    qx = 0.25 * S
    qy = (m01 + m10) / S
    qz = (m02 + m20) / S
  elif m11 > m22:
    S = sqrt(1.0 + m11 - m00 - m22) * 2.0
    # S=4*qy
    qw = (m02 - m20) / S
    qx = (m01 + m10) / S
    qy = 0.25 * S
    qz = (m12 + m21) / S
  else:
    S = sqrt(1.0 + m22 - m00 - m11) * 2.0
    # S=4*qz
    qw = (m10 - m01) / S
    qx = (m02 + m20) / S
    qy = (m12 + m21) / S
    qz = 0.25 * S

  return quat_normalize(np.array([qw, qx, qy, qz]))


class PID:
  """ PID controller """
  def __init__(self, k_p, k_i, k_d):
    self.k_p = k_p
    self.k_i = k_i
    self.k_d = k_d

    self.error_p = 0.0
    self.error_i = 0.0
    self.error_d = 0.0
    self.error_prev = 0.0
    self.error_sum = 0.0

  def update(self, setpoint, actual, dt):
    """ Update """
    # Calculate errors
    error = setpoint - actual
    self.error_sum += error * dt

    # Calculate output
    self.error_p = self.k_p * error
    self.error_i = self.k_i * self.error_sum
    self.error_d = self.k_d * (error - self.error_prev) / dt
    output = self.error_p + self.error_i + self.error_d

    # Keep track of error
    self.error_prev = error

    return output

  def reset(self):
    """ Reset """
    self.error_prev = 0
    self.error_sum = 0
    self.error_p = 0
    self.error_i = 0
    self.error_d = 0


def plot_mocap_filter(filter_csv):
  """ Plot filter data """
  mocap_data = pandas.read_csv(filter_csv)
  mocap_time = (mocap_data["#ts"] - mocap_data["#ts"][0]) * 1e-9
  pos_est = mocap_data[["pos_est_x", "pos_est_y", "pos_est_z"]].to_numpy()
  vel_est = mocap_data[["vel_est_x", "vel_est_y", "vel_est_z"]].to_numpy()
  pos_gnd = mocap_data[["pos_gnd_x", "pos_gnd_y", "pos_gnd_z"]].to_numpy()

  # Plot position
  plt.figure()
  plt.subplot(311)
  plt.plot(mocap_time, pos_est[:, 0], "r-")
  plt.plot(mocap_time, pos_gnd[:, 0], "k--")
  plt.xlabel("Time [s]")
  plt.ylabel("Displacement [m]")

  plt.subplot(312)
  plt.plot(mocap_time, pos_est[:, 1], "g-")
  plt.plot(mocap_time, pos_gnd[:, 1], "k--")
  plt.xlabel("Time [s]")
  plt.ylabel("Displacement [m]")

  plt.subplot(313)
  plt.plot(mocap_time, pos_est[:, 2], "b-")
  plt.plot(mocap_time, pos_gnd[:, 2], "k--")
  plt.xlabel("Time [s]")
  plt.ylabel("Displacement [m]")

  # Plot velocity
  plt.figure()
  plt.subplot(311)
  plt.plot(mocap_time, vel_est[:, 0], "r-")
  plt.xlabel("Time [s]")
  plt.ylabel("Displacement [m]")

  plt.subplot(312)
  plt.plot(mocap_time, vel_est[:, 1], "g-")
  plt.xlabel("Time [s]")
  plt.ylabel("Displacement [m]")

  plt.subplot(313)
  plt.plot(mocap_time, vel_est[:, 2], "b-")
  plt.xlabel("Time [s]")
  plt.ylabel("Displacement [m]")

  plt.show()


class MavVelocityControl:
  def __init__(self):
    self.period = 0.0049  # [s]
    self.dt = 0
    self.pid_vx = PID(1.0, 0.0, 0.05)
    self.pid_vy = PID(1.0, 0.0, 0.05)
    self.pid_vz = PID(1.0, 0.0, 0.05)
    self.hover_thrust = 0.7
    self.u = [0.0, 0.0, 0.0, 0.0]  # roll, pitch, yaw, thrust

  def update(self, sp, pv, dt):
    """ Update """
    # Check rate
    self.dt += dt
    if self.dt < self.period:
      return self.u  # Return previous command

    # Transform errors in world frame to mav frame
    errors_W = np.array([sp[0] - pv[0], sp[1] - pv[1], sp[2] - pv[2]])
    C_WB = euler321(pv[3], 0.0, 0.0)
    errors = C_WB.T @ errors_W

    # Roll, pitch, yaw and thrust
    r = -self.pid_vy.update(errors[1], 0.0, dt)
    p = self.pid_vx.update(errors[0], 0.0, dt)
    y = sp[3]
    t = self.hover_thrust + self.pid_vz.update(errors[2], 0.0, dt)

    # Clip values
    self.u[0] = clip_value(r, deg2rad(-20.0), deg2rad(20.0))
    self.u[1] = clip_value(p, deg2rad(-20.0), deg2rad(20.0))
    self.u[2] = y
    self.u[3] = clip_value(t, 0.0, 1.0)

    # Reset dt
    self.dt = 0.0

    return self.u

  def reset(self):
    """ Reset """
    self.dt = 0.0
    self.pid_vx.reset()
    self.pid_vy.reset()
    self.pid_vz.reset()
    self.u = [0.0, 0.0, 0.0, 0.0]


class MavPositionControl:
  def __init__(self, output_mode="VELOCITY"):
    self.output_mode = output_mode
    self.dt = 0
    self.u = [0.0, 0.0, 0.0, 0.0]

    if self.output_mode == "VELOCITY":
      self.period = 0.0055
      self.vx_min = -5.0
      self.vx_max = 5.0
      self.vy_min = -5.0
      self.vy_max = 5.0
      self.vz_min = -5.0
      self.vz_max = 5.0

      self.dt = 0
      self.pid_x = PID(0.5, 0.0, 0.05)
      self.pid_y = PID(0.5, 0.0, 0.05)
      self.pid_z = PID(0.5, 0.0, 0.05)

    elif self.output_mode == "ATTITUDE":
      self.period = 0.0049
      self.roll_min = deg2rad(-35.0)
      self.roll_max = deg2rad(35.0)
      self.pitch_min = deg2rad(-35.0)
      self.pitch_max = deg2rad(35.0)
      self.hover_thrust = 0.3

      self.pid_x = PID(1.0, 0.0, 0.1)
      self.pid_y = PID(1.0, 0.0, 0.1)
      self.pid_z = PID(0.1, 0.0, 0.0)

    else:
      raise NotImplementedError()

  def update(self, sp, pv, dt):
    """ Update """
    # Check rate
    self.dt += dt
    if self.dt < self.period:
      return self.u  # Return previous command

    if self.output_mode == "VELOCITY":
      # Calculate velocity errors in world frame
      errors = np.array([sp[0] - pv[0], sp[1] - pv[1], sp[2] - pv[2]])

      # Velocity commands
      vx = self.pid_x.update(errors[0], 0.0, self.dt)
      vy = self.pid_y.update(errors[1], 0.0, self.dt)
      vz = self.pid_z.update(errors[2], 0.0, self.dt)
      yaw = sp[3]

      self.u[0] = clip_value(vx, self.vx_min, self.vx_max)
      self.u[1] = clip_value(vy, self.vy_min, self.vy_max)
      self.u[2] = clip_value(vz, self.vz_min, self.vz_max)
      self.u[3] = yaw

    elif self.output_mode == "ATTITUDE":
      # Calculate position errors in mav frame
      errors = euler321(pv[3], 0.0, 0.0).T @ (sp[0:3] - pv[0:3])

      # Attitude commands
      roll = -self.pid_y.update(errors[1], 0.0, dt)
      pitch = self.pid_x.update(errors[0], 0.0, dt)
      thrust = self.pid_z.update(errors[2], 0.0, dt)

      # Attitude command (roll, pitch, yaw, thrust)
      self.u[0] = clip_value(roll, self.roll_min, self.roll_max)
      self.u[1] = clip_value(pitch, self.pitch_min, self.pitch_max)
      self.u[2] = sp[3]
      self.u[3] = clip_value(thrust, 0.0, 1.0)

    else:
      raise NotImplementedError()

    # Reset dt
    self.dt = 0.0

    return self.u

  def reset(self):
    """ Reset """
    self.dt = 0.0
    self.pid_vx.reset()
    self.pid_vy.reset()
    self.pid_vz.reset()
    self.u = [0.0, 0.0, 0.0, 0.0]


class MavTrajectoryControl:
  def __init__(self, **kwargs):
    self.A = kwargs.get("A", 2.0)
    self.B = kwargs.get("B", 2.0)
    self.a = kwargs.get("a", 1.0)
    self.b = kwargs.get("b", 2.0)
    self.z = kwargs["z"]
    self.T = kwargs["T"]
    self.f = 1.0 / self.T
    self.delta = kwargs.get("delta", np.pi)

    # Position and velocity controller
    self.last_ts = None
    self.pos_ctrl = MavPositionControl("ATTITUDE")
    self.vel_ctrl = MavVelocityControl()

  def get_traj(self):
    """ Return trajectory """
    pos_data = np.zeros((3, 1000))
    time = np.linspace(0.0, self.T, 1000)
    for i, t in enumerate(time):
      pos_data[:, i] = self.get_position(t).T
    return pos_data.T

  def get_position(self, t):
    """ Get position """
    w = 2.0 * np.pi * self.f
    theta = np.sin(0.25 * w * t)**2

    ka = 2.0 * np.pi * self.a
    kb = 2.0 * np.pi * self.b

    x = self.A * np.sin(ka * theta + self.delta)
    y = self.B * np.sin(kb * theta)
    z = self.z

    return np.array([x, y, z])

  def get_yaw(self, t):
    """ Get yaw """
    p0 = self.get_position(t)
    p1 = self.get_position(t + 0.1)
    dx, dy, dz = p1 - p0

    heading = np.arctan2(dy, dx)
    if heading > np.pi:
      heading -= 2.0 * np.pi
    elif heading < -np.pi:
      heading += 2.0 * np.pi

    return heading

  def get_velocity(self, t):
    w = 2.0 * np.pi * self.f
    theta = np.sin(0.25 * w * t)**2

    ka = 2.0 * np.pi * self.a
    kb = 2.0 * np.pi * self.b
    kpift = 0.5 * np.pi * self.f * t
    kx = 2.0 * np.pi**2 * self.A * self.a * self.f
    ky = 2.0 * np.pi**2 * self.B * self.b * self.f
    ksincos = np.sin(kpift) * np.cos(kpift)

    vx = kx * ksincos * np.cos(ka * np.sin(kpift)**2 + self.delta)
    vy = ky * ksincos * np.cos(kb * np.sin(kpift)**2)
    vz = 0.0

    return np.array([vx, vy, vz])

  def update(self, t, pos_pv, vel_pv):
    # Pre-check
    if self.last_ts is None:
      self.last_ts = t
      return np.array([0.0, 0.0, 0.0, 0.0])
    # dt = t - self.last_ts
    dt = 0.005

    # Get trajectory position, velocity and yaw
    traj_pos = self.get_position(t)
    traj_vel = self.get_velocity(t)
    traj_yaw = self.get_yaw(t)

    # Form position and velocity setpoints
    pos_sp = np.array([traj_pos[0], traj_pos[1], traj_pos[2], traj_yaw])
    vel_sp = [traj_vel[0], traj_vel[1], traj_vel[2], traj_yaw]

    # Position control
    att_pos_sp = self.pos_ctrl.update(pos_sp, pos_pv, dt)

    # Velocity control
    att_vel_sp = self.vel_ctrl.update(vel_sp, vel_pv, dt)

    # Mix both position and velocity control into a single attitude setpoint
    att_sp = np.array([0.0, 0.0, 0.0, 0.0])
    att_sp[0] = att_vel_sp[0] + att_pos_sp[0]
    att_sp[1] = att_vel_sp[1] + att_pos_sp[1]
    att_sp[2] = traj_yaw
    att_sp[3] = att_vel_sp[3] + att_pos_sp[3]

    att_sp[0] = clip_value(att_sp[0], deg2rad(-35.0), deg2rad(35.0))
    att_sp[1] = clip_value(att_sp[1], deg2rad(-35.0), deg2rad(35.0))
    att_sp[2] = att_sp[2]
    att_sp[3] = clip_value(att_sp[3], 0.0, 1.0)

    # Update
    self.last_ts = t

    return att_sp

  def plot_traj(self):
    """ Plot """
    pos_data = np.zeros((3, 1000))
    vel_data = np.zeros((3, 1000))
    time = np.linspace(0.0, self.T, 1000)
    for i, t in enumerate(time):
      pos_data[:, i] = self.get_position(t).T
      vel_data[:, i] = self.get_velocity(t).T

    plt.subplot(311)
    plt.plot(pos_data[0, :], pos_data[1, :])
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    plt.subplot(312)
    plt.plot(time, pos_data[0, :], "r-", label="Position - x")
    plt.plot(time, pos_data[1, :], "g-", label="Position - y")
    plt.plot(time, pos_data[2, :], "b-", label="Position - z")
    plt.xlabel("Time [s]")
    plt.ylabel("Positions [m]")
    plt.legend(loc=0)

    plt.subplot(313)
    plt.plot(time, vel_data[0, :], "r-", label="Velocity - x")
    plt.plot(time, vel_data[1, :], "g-", label="Velocity - y")
    plt.plot(time, vel_data[2, :], "b-", label="Velocity - z")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [ms^-1]")
    plt.legend(loc=0)

    plt.show()


class MavMode(Enum):
  START = 1
  TUNE = 2
  TRAJ = 3
  HOVER = 4
  LAND = 5


class MocapFilter:
  def __init__(self):
    self.initialized = False

    # Initial state x
    self.x = np.zeros(9)

    # Convariance
    self.P = np.eye(9)

    # Measurement Matrix
    self.H = np.block([np.eye(3), np.zeros((3, 6))])

    # Process Noise Matrix
    self.Q = np.eye(9)
    self.Q[0:3, 0:3] = 0.01 * np.eye(3)
    self.Q[3:6, 3:6] = 0.00001**2 * np.eye(3)
    self.Q[6:9, 6:9] = 0.00001**2 * np.eye(3)

    # Measurement Noise Matrix
    self.R = 0.1**2 * np.eye(3)

  def get_position(self):
    """ Get Position """
    return np.array([self.x[0], self.x[1], self.x[2]])

  def get_velocity(self):
    """ Get Velocity """
    return np.array([self.x[3], self.x[4], self.x[5]])

  def update(self, z, dt):
    """ Update """
    # Initialize
    if self.initialized is False:
      self.x[0] = z[0]
      self.x[1] = z[1]
      self.x[2] = z[2]
      self.initialized = True
      return False

    # Predict
    # -- Transition Matrix
    F = np.eye(9)
    F[0:3, 3:6] = np.eye(3) * dt
    F[0:3, 6:9] = np.eye(3) * dt**2
    F[3:6, 6:9] = np.eye(3) * dt
    # -- Predict
    self.x = F @ self.x
    self.P = F @ self.P @ F.T + self.Q

    # Update
    I = np.eye(9)
    y = z - self.H @ self.x
    S = self.R + self.H @ self.P @ self.H.T
    K = self.P @ self.H.T @ np.linalg.inv(S)
    self.x = self.x + K @ y
    self.P = (I - K @ self.H) @ self.P

    return True


def set_px4_param(param_name, param_value):
  rospy.wait_for_service('/mavros/param/set')
  try:
    param_set = rospy.ServiceProxy('/mavros/param/set', ParamSet)
    param_value_msg = ParamValue()
    param_value_msg.integer = param_value
    response = param_set(param_name, param_value_msg)

    if response.success:
      rospy.loginfo("Parameter %s set to %s", param_name, param_value)
    else:
      rospy.logerr("Failed to set parameter %s", param_name)

  except rospy.ServiceException as e:
    rospy.logerr("Service call failed: %s", e)


class MavNode:
  """Node for controlling a vehicle in offboard mode."""
  def __init__(self, **kwargs):
    self.sim_mode = kwargs.get("sim_mode", True)

    self.is_running = True
    topic_mode = "mavros/set_mode"
    topic_arming = "mavros/cmd/arming"
    topic_state = "/mavros/state"
    topic_param_set = "/mavros/param/set"
    topic_pose = "/mavros/local_position/pose"
    topic_pos_set = "/mavros/setpoint_position/local"
    topic_att_set = "/mavros/setpoint_attitude/attitude"
    topic_thr_set = "/mavros/setpoint_attitude/thrust"
    topic_mocap = "/vicon/srl_mav/srl_mav"

    # Create service proxies
    self.arming_client = rospy.ServiceProxy(topic_arming, CommandBool)
    self.mode_client = rospy.ServiceProxy(topic_mode, SetMode)
    rospy.wait_for_service(topic_arming)
    rospy.wait_for_service(topic_mode)

    # Create subscribers
    self.sub_state = rospy.Subscriber(topic_state, State, self.state_cb)
    self.sub_pose = rospy.Subscriber(topic_pose, PoseStamped, self.pose_cb)
    # self.sub_mocap = sub_init(PoseStamped, topic_mocap, self.mocap_cb, 1)

    # Create publishers
    self.pub_pos_set = rospy.Publisher(topic_pos_set, PoseStamped)
    self.pub_att_set = rospy.Publisher(topic_att_set, PoseStamped)
    self.pub_thr_set = rospy.Publisher(topic_thr_set, Thrust)

    # State
    self.local_pose = None
    self.state = None
    self.mode = MavMode.START
    self.ts = None
    self.ts_prev = None
    self.pos = None
    self.vel = None
    self.heading = 0.0
    self.traj_start = None
    self.hover_start = None
    self.tune_start = None

    # Settings
    self.takeoff_height = 2.0
    self.traj_period = 30.0
    self.hover_for = 3.0

    # Filter
    self.filter = MocapFilter()
    self.mocap_csv = open("/tmp/mocap_filter.csv", "w")
    self.mocap_csv.write("#ts,")
    self.mocap_csv.write("pos_est_x,pos_est_y,pos_est_z,")
    self.mocap_csv.write("vel_est_x,vel_est_y,vel_est_z,")
    self.mocap_csv.write("pos_gnd_x,pos_gnd_y,pos_gnd_z\n")

    # Control
    self.pos_ctrl = MavPositionControl()
    self.vel_ctrl = MavVelocityControl()
    self.traj_ctrl = MavTrajectoryControl(a=1,
                                          b=2,
                                          delta=np.pi / 2,
                                          z=self.takeoff_height,
                                          T=self.traj_period)
    self.yaw_sp = 0.0
    self.pos_sp = [0.0, 0.0, 2.0, 0.0]

    # Create a timer to publish control commands
    self.dt = 0.005
    # self.timer = self.create_timer(self.dt, self.timer_cb)

    # Wait until we are connected to the FCU
    rate = rospy.Rate(10)
    while self.state is None or not self.state.connected:
      rate.sleep()
    rospy.loginfo("Connected!")

    # Engage offboard and arm MAV if in sim mode
    if self.sim_mode:
      set_px4_param("COM_RCL_EXCEPT", 4)
      set_px4_param("NAV_DLL_ACT", 0)
      set_px4_param("NAV_RCL_ACT", 0)
      self.engage_offboard_mode()
      self.arm()

    # Position control tuning waypoints
    self.vel_tune_setpoints = [
        # Tune z-axis
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, +0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        # Tune x-axis
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        # Tune y-axis
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ]

    # Position control tuning waypoints
    self.pos_tune_setpoints = [
        # Tune x-axis
        [0.0, 0.0, self.takeoff_height, 0.0],
        [1.0, 0.0, self.takeoff_height, 0.0],
        [-1.0, 0.0, self.takeoff_height, 0.0],
        [0.0, 0.0, self.takeoff_height, 0.0],
        # Tune y-axis
        [0.0, 0.0, self.takeoff_height, 0.0],
        [0.0, 0.0, self.takeoff_height, 0.0],
        [0.0, 1.0, self.takeoff_height, 0.0],
        [0.0, -1.0, self.takeoff_height, 0.0],
        [0.0, 0.0, self.takeoff_height, 0.0]
    ]

    # Data
    self.pos_time = []
    self.pos_actual = []
    self.pos_traj = []

  def __del__(self):
    """ Destructor """
    if self.mocap_csv:
      self.mocap_csv.close()

  def pose_cb(self, msg):
    """Callback function for msg topic subscriber."""
    self.local_pose = msg
    self.ts = msg.header.stamp.to_nsec()

    px = msg.pose.position.x
    py = msg.pose.position.y
    pz = msg.pose.position.z
    self.pos = np.array([px, py, pz])

    qw = msg.pose.orientation.w
    qx = msg.pose.orientation.x
    qy = msg.pose.orientation.y
    qz = msg.pose.orientation.z
    q = np.array([qw, qx, qy, qz])
    ypr = quat2euler(q)

    self.heading = ypr[0]
    if self.heading > np.pi:
      self.heading -= 2.0 * np.pi
    elif self.heading < -np.pi:
      self.heading += 2.0 * np.pi

  def mocap_cb(self, msg):
    """Mocap callback"""
    self.ts = msg.header.stamp.to_nsec()
    dt = float((self.ts - self.ts_prev) * 1e-9) if self.ts_prev else 0.0
    self.ts_prev = self.ts

    rx = msg.pose.position.x
    ry = msg.pose.position.y
    rz = msg.pose.position.z
    qw = msg.pose.orientation.w
    qx = msg.pose.orientation.x
    qy = msg.pose.orientation.y
    qz = msg.pose.orientation.z
    pos = np.array([rx, ry, rz])
    quat = np.array([qw, qx, qy, qz])

    if self.filter.update(pos, dt):
      self.pos = self.filter.get_position()
      self.vel = self.filter.get_velocity()
      self.mocap_pos = pos
      self.mocap_quat = quat

  # def pos_sp_cb(self, msg):
  #   """Callback function for position setpoint topic subscriber."""
  #   self.pos_sp = [msg.x, msg.y, msg.z, self.yaw_sp]

  # def yaw_sp_cb(self, msg):
  #   """Callback function for yaw setpoint topic subscriber."""
  #   self.yaw_sp = msg.data
  #   self.pos_sp[3] = self.yaw_sp

  def state_cb(self, msg):
    """Callback function for vehicle_state topic subscriber."""
    self.state = msg

  def is_armed(self):
    """ Is armed? """
    return self.state.armed

  def is_offboard(self):
    """ Is offboard? """
    return self.state.mode == "OFFBOARD"

  def arm(self):
    """Send an arm command to the vehicle."""
    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True
    self.arming_client.call(arm_cmd)
    rospy.loginfo("Arming!")

  def disarm(self):
    """Send a disarm command to the vehicle."""
    arm_cmd = CommandBoolRequest()
    arm_cmd.value = False
    self.arming_client.call(arm_cmd)
    rospy.loginfo("Disarming!")

  def engage_offboard_mode(self):
    """Switch to offboard mode."""
    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'
    self.mode_client.call(offb_set_mode)
    rospy.loginfo("Offboard mode!")
    time.sleep(1)

  def land(self):
    """Switch to land mode."""
    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'LAND'
    self.mode_client.call(offb_set_mode)
    rospy.loginfo("Land!")

  def pub_attitude_sp(self, roll, pitch, yaw, thrust):
    yaw_sp = deg2rad(90.0) - yaw
    if yaw_sp >= np.pi:
      yaw_sp -= 2.0 * np.pi
    elif yaw_sp <= -np.pi:
      yaw_sp += 2.0 * np.pi

    qw, qx, qy, qz = quat_normalize(euler2quat(yaw_sp, pitch, roll)).tolist()
    msg = PoseStamped()
    msg.pose.position.x = 0.0
    msg.pose.position.y = 0.0
    msg.pose.position.z = 0.0
    msg.pose.orientation.w = qw
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    self.pub_att.publish(msg)

    msg = Thrust()
    msg.thrust = thrust
    self.pub_thr.publish(msg)

  def execute_velocity_control_test(self):
    # Process variables
    pos_pv = [self.pos[0], self.pos[1], self.pos[2], self.heading]
    vel_pv = [self.vel[0], self.vel[1], self.vel[2], self.heading]

    if self.mode == MavMode.START:
      # Start hover timer
      if self.hover_start is None:
        self.hover_start = self.ts

      # Get hover point
      self.pos_sp = [0.0, 0.0, self.takeoff_height, self.yaw_sp]

      # Update position controller
      vel_sp = self.pos_ctrl.update(self.pos_sp, pos_pv, self.dt)
      u = self.vel_ctrl.update(vel_sp, vel_pv, self.dt)
      self.pub_attitude_sp(u[0], u[1], u[2], u[3])

      # Transition to land?
      hover_time = float(self.ts - self.hover_start) * 1e-9
      dx = self.pos[0] - self.pos_sp[0]
      dy = self.pos[1] - self.pos_sp[1]
      dz = self.pos[2] - self.pos_sp[2]
      dpos = np.sqrt(dx * dx + dy * dy + dz * dz)
      dyaw = np.fabs(self.heading - self.yaw_sp)
      if dpos < 0.2 and dyaw < np.deg2rad(10.0) and hover_time > 3.0:
        self.get_logger().info('TRANSITION TO TUNE!')
        self.mode = MavMode.TUNE
        self.hover_start = None

    elif self.mode == MavMode.TUNE:
      # Start hover timer
      if self.tune_start is None:
        self.tune_start = self.ts

      # Velocity controller
      vel_sp = self.vel_tune_setpoints[0]
      u = self.vel_ctrl.update(vel_sp, vel_pv, self.dt)
      self.pub_attitude_sp(u[0], u[1], u[2], u[3])

      # Check if position setpoint reached
      tune_time = float(self.ts - self.tune_start) * 1e-9
      if tune_time >= 2.0:
        self.get_logger().info('BACK TO START!')
        self.mode = MavMode.START
        self.vel_tune_setpoints.pop(0)
        self.tune_start = None
        self.hover_start = None

      # Land?
      if len(self.vel_tune_setpoints) == 0:
        self.mode = MavMode.LAND
        self.tune_start = None

    elif self.mode == MavMode.LAND:
      # Dis-armed?
      if self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
        self.stop_node()
        return

      # Land
      if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
        self.land()

  # def execute_position_control_test(self):
  #   # Start tune timestamp
  #   if self.tune_start is None:
  #     self.tune_start = self.ts

  #   # Process variables
  #   pos_pv = [self.pos[0], self.pos[1], self.pos[2], self.heading]
  #   vel_pv = [self.vel[0], self.vel[1], self.vel[2], self.heading]

  #   if self.mode == MavMode.START:
  #     # Start hover timer
  #     if self.hover_start is None:
  #       self.hover_start = self.ts

  #     # Get hover point
  #     self.pos_sp = [0.0, 0.0, self.takeoff_height, self.yaw_sp]

  #     # Update position controller
  #     vel_sp = self.pos_ctrl.update(self.pos_sp, pos_pv, self.dt)
  #     u = self.vel_ctrl.update(vel_sp, vel_pv, self.dt)
  #     self.pub_attitude_sp(u[0], u[1], u[2], u[3])

  #     # Transition to land?
  #     hover_time = float(self.ts - self.hover_start) * 1e-9
  #     dx = self.pos[0] - self.pos_sp[0]
  #     dy = self.pos[1] - self.pos_sp[1]
  #     dz = self.pos[2] - self.pos_sp[2]
  #     dpos = np.sqrt(dx * dx + dy * dy + dz * dz)
  #     dyaw = np.fabs(self.heading - self.yaw_sp)
  #     if dpos < 0.2 and dyaw < np.deg2rad(10.0) and hover_time > 3.0:
  #       self.get_logger().info('TRANSITION TO TUNE!')
  #       self.mode = MavMode.TUNE
  #       self.hover_start = None

  #   elif self.mode == MavMode.TUNE:
  #     # Position controller
  #     self.pos_sp = self.pos_tune_setpoints[0]
  #     vel_sp = self.pos_ctrl.update(self.pos_sp, pos_pv, self.dt)
  #     u = self.vel_ctrl.update(vel_sp, vel_pv, self.dt)
  #     self.pub_attitude_sp(u[0], u[1], u[2], u[3])

  #     # Check if position setpoint reached
  #     tune_time = float(self.ts - self.tune_start) * 1e-9
  #     dx = self.pos[0] - self.pos_sp[0]
  #     dy = self.pos[1] - self.pos_sp[1]
  #     dz = self.pos[2] - self.pos_sp[2]
  #     dpos = np.sqrt(dx * dx + dy * dy + dz * dz)
  #     if dpos < 0.1 and tune_time >= self.hover_for:
  #       self.pos_tune_setpoints.pop(0)
  #       self.tune_start = None

  #     # Land?
  #     if len(self.pos_tune_setpoints) == 0:
  #       self.mode = MavMode.LAND
  #       self.tune_start = None

  #   elif self.mode == MavMode.LAND:
  #     # Dis-armed?
  #     if self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
  #       self.stop_node()
  #       return

  #     # Land
  #     if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
  #       self.land()

  # def execute_trajectory(self):
  #   # Process variables
  #   pos_pv = [self.pos[0], self.pos[1], self.pos[2], self.heading]
  #   vel_pv = [self.vel[0], self.vel[1], self.vel[2], self.heading]

  #   # GO TO START
  #   if self.mode == MavMode.START:
  #     # Set yaw and position setpoint
  #     x0, y0, z0 = self.traj_ctrl.get_position(0.0)
  #     self.yaw_sp = self.traj_ctrl.get_yaw(0.0)
  #     self.pos_sp = [x0, y0, z0, self.yaw_sp]

  #     # Position controller
  #     vel_sp = self.pos_ctrl.update(self.pos_sp, pos_pv, self.dt)
  #     u = self.vel_ctrl.update(vel_sp, vel_pv, self.dt)
  #     self.pub_attitude_sp(u[0], u[1], u[2], u[3])

  #     # Transition to trajectory mode?
  #     dx = self.pos[0] - self.pos_sp[0]
  #     dy = self.pos[1] - self.pos_sp[1]
  #     dz = self.pos[2] - self.pos_sp[2]
  #     dpos = np.sqrt(dx * dx + dy * dy + dz * dz)
  #     dyaw = np.fabs(self.heading - self.yaw_sp)
  #     if dpos < 0.1 and dyaw < np.deg2rad(10.0):
  #       self.mode = MavMode.TRAJ

  #   # EXECUTE TRAJECTORY
  #   elif self.mode == MavMode.TRAJ:
  #     # Run trajectory
  #     if self.traj_start is None:
  #       self.traj_start = self.ts

  #     # Update trajectory controller
  #     t = float(self.ts - self.traj_start) * 1e-9
  #     u = self.traj_ctrl.update(t, pos_pv, vel_pv)
  #     self.pub_attitude_sp(u[0], u[1], u[2], u[3])

  #     # Record position
  #     self.pos_time.append(t)
  #     self.pos_actual.append([self.pos[0], self.pos[1], self.pos[2]])
  #     self.pos_traj.append(self.traj_ctrl.get_position(t))

  #     # Transition to hover?
  #     if t >= self.traj_period:
  #       self.mode = MavMode.HOVER
  #       self.traj_start = None

  #   # HOVER
  #   elif self.mode == MavMode.HOVER:
  #     # Start hover timer
  #     if self.hover_start is None:
  #       self.hover_start = self.ts

  #     # Get hover point
  #     pos = self.traj_ctrl.get_position(0.0)
  #     yaw = self.traj_ctrl.get_yaw(0.0)
  #     self.pos_sp = [pos[0], pos[1], pos[2], yaw]

  #     # Update position controller
  #     vel_sp = self.pos_ctrl.update(self.pos_sp, pos_pv, self.dt)
  #     u = self.vel_ctrl.update(vel_sp, vel_pv, self.dt)
  #     self.pub_attitude_sp(u[0], u[1], u[2], u[3])

  #     # Transition to land?
  #     hover_time = float(self.ts - self.hover_start) * 1e-9
  #     if hover_time > self.hover_for:
  #       self.hover_start = None
  #       self.mode = MavMode.LAND

  #   # LAND
  #   elif self.mode == MavMode.LAND:
  #     # Dis-armed?
  #     if self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
  #       self.stop_node()
  #       return

  #     # Land
  #     if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
  #       self.land()

  # def mocap_record(self, ts, pos_est, vel_est, pos_gnd):
  #   """ Record mocap filter """
  #   self.mocap_csv.write(f"{ts},")
  #   self.mocap_csv.write(f"{pos_est[0]},{pos_est[1]},{pos_est[2]},")
  #   self.mocap_csv.write(f"{vel_est[0]},{vel_est[1]},{vel_est[2]},")
  #   self.mocap_csv.write(f"{pos_gnd[0]},{pos_gnd[1]},{pos_gnd[2]}\n")

  # def timer_cb(self):
  #   """Callback function for the timer."""
  #   # Check we are receiving position and velocity information
  #   if self.pos is None or self.vel is None:
  #     return

  #   # Check MAV is armed and offboard mode activated
  #   if not self.is_armed():
  #     return
  #   if not self.is_offboard():
  #     return

  #   # self.mocap_record(self.ts, self.pos, self.vel, self.mocap_pos)

  #   # self.execute_velocity_control_test()
  #   # self.execute_position_control_test()
  #   self.execute_trajectory()

  # def stop_node(self):
  #   """ Stop Node """
  #   self.get_logger().info('Stopping the node')
  #   self.timer.cancel()
  #   self.destroy_timer(self.timer)

  #   self.sub_pos.destroy()
  #   self.sub_pos_sp.destroy()
  #   self.sub_yaw_sp.destroy()
  #   self.sub_status.destroy()

  #   self.pos_time = np.array(self.pos_time)
  #   self.pos_actual = np.array(self.pos_actual)
  #   self.pos_traj = np.array(self.pos_traj)

  #   plt.plot(self.pos_actual[:, 0], self.pos_actual[:, 1], "r-", label="Actual")
  #   plt.plot(self.pos_traj[:, 0],
  #            self.pos_traj[:, 1],
  #            "k--",
  #            label="Trajectory")
  #   plt.show()

  #   self.get_logger().info('Destroying the node')
  #   self.destroy_node()
  #   self.is_running = False


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--sim_mode", default=True)
  args = parser.parse_args()

  # Run Mav Node
  rospy.init_node("mav_node")
  node = MavNode(sim_mode=args.sim_mode)
  rospy.loginfo("Sending position setpoint!")

  # rate = rospy.Rate(20)
  # pose = PoseStamped()
  # pose.pose.position.x = 0
  # pose.pose.position.y = 0
  # pose.pose.position.z = 2
  # while (not rospy.is_shutdown()):
  #   node.pub_pos_set.publish(pose)
  #   rate.sleep()
  node.execute_velocity_control_test()
  rospy.spin()
