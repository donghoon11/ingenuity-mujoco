"""
Ingenuity Mars Helicopter - PID Controllers

Altitude PID and Attitude PD controllers tuned for Mars environment.
Designed for use with mhs_mars.xml (gear=6, gravity=-3.71).
"""

import numpy as np

from config import (
    HOVER_CTRL,
    ALT_KP, ALT_KI, ALT_KD, ALT_I_LIMIT,
    ATT_KP, ATT_KD,
)


class AltitudePID:
    """
    PID altitude controller with anti-windup.

    Output is thrust ctrl value [0, 1] for each rotor.
    Includes feedforward term for hover equilibrium.
    """

    def __init__(self, kp=ALT_KP, ki=ALT_KI, kd=ALT_KD,
                 i_limit=ALT_I_LIMIT, ff_thrust=HOVER_CTRL):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.ff_thrust = ff_thrust

        self.integral = 0.0
        self.err_prev = 0.0

    def compute(self, z_ref: float, z_meas: float, dt: float) -> float:
        """
        Compute thrust ctrl command.

        Parameters
        ----------
        z_ref  : Target altitude (m)
        z_meas : Measured altitude (m)
        dt     : Timestep (s)

        Returns
        -------
        thrust_ctrl : float in [0, 1]
        """
        err = z_ref - z_meas

        self.integral += err * dt
        self.integral = np.clip(self.integral, -self.i_limit, self.i_limit)

        derivative = (err - self.err_prev) / dt if dt > 0 else 0.0
        self.err_prev = err

        output = self.ff_thrust + self.kp * err + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, 0.0, 1.0))

    def reset(self):
        self.integral = 0.0
        self.err_prev = 0.0


class AttitudePD:
    """
    PD attitude controller for roll/pitch/yaw stabilization.

    Output is ctrl value [-1, 1] for x_movement, y_movement, z_rotation.
    """

    def __init__(self, kp=ATT_KP, kd=ATT_KD):
        self.kp = kp
        self.kd = kd

    def compute(self, roll_ref: float, pitch_ref: float, yaw_ref: float,
                roll: float, pitch: float, yaw: float,
                p: float, q: float, r: float) -> tuple:
        """
        Compute attitude control commands.

        Parameters
        ----------
        roll_ref, pitch_ref, yaw_ref : Target angles (rad)
        roll, pitch, yaw             : Measured angles (rad)
        p, q, r                      : Angular velocities (rad/s)

        Returns
        -------
        x_cmd, y_cmd, z_cmd : float, each in [-1, 1]
        """
        roll_err = roll_ref - roll
        pitch_err = pitch_ref - pitch
        yaw_err = yaw_ref - yaw

        # Wrap yaw error to [-pi, pi]
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi

        x_cmd = self.kp * roll_err - self.kd * p
        y_cmd = self.kp * pitch_err - self.kd * q
        z_cmd = self.kp * yaw_err - self.kd * r

        x_cmd = float(np.clip(x_cmd, -1.0, 1.0))
        y_cmd = float(np.clip(y_cmd, -1.0, 1.0))
        z_cmd = float(np.clip(z_cmd, -1.0, 1.0))

        return x_cmd, y_cmd, z_cmd


class ForwardFlightController:
    """
    Outer-loop position/velocity controller.
    Converts position errors to roll/pitch reference angles.
    """

    def __init__(self, kp_pos=0.3, kd_pos=0.5,
                 max_tilt=np.radians(15)):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.max_tilt = max_tilt

    def compute(self, pos_ref: np.ndarray, pos: np.ndarray,
                vel: np.ndarray) -> tuple:
        """
        Compute roll/pitch references from position error.

        Parameters
        ----------
        pos_ref : [x_ref, y_ref, z_ref]
        pos     : [x, y, z]
        vel     : [vx, vy, vz]

        Returns
        -------
        roll_ref, pitch_ref : float (rad)
        """
        err_x = pos_ref[0] - pos[0]
        err_y = pos_ref[1] - pos[1]

        # Desired pitch for X motion, roll for Y motion
        pitch_ref = self.kp_pos * err_x - self.kd_pos * vel[0]
        roll_ref = -(self.kp_pos * err_y - self.kd_pos * vel[1])

        pitch_ref = float(np.clip(pitch_ref, -self.max_tilt, self.max_tilt))
        roll_ref = float(np.clip(roll_ref, -self.max_tilt, self.max_tilt))

        return roll_ref, pitch_ref
