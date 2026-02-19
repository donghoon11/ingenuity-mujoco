"""
Ingenuity Mars Helicopter - MuJoCo Simulation Interface

Bridge between BEMT aerodynamic model and MuJoCo dynamics.
Handles model loading, simulation execution, and KPI extraction.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from config import (
    SCENE_MARS_XML, DT, Z_REF, SIM_DURATION, INIT_POS_Z,
    IDX_THRUST1, IDX_THRUST2, IDX_ROLL, IDX_PITCH, IDX_YAW,
    HOVER_CTRL, THRUST_GEAR,
)
from utils import (
    load_mars_model, get_body_id, extract_state, reset_model,
    quat_to_euler, get_sensor_data, get_rotor_joint_indices,
    update_rotor_visual, update_tracking_camera,
)
from controller import AltitudePID, AttitudePD


class SimResult:
    """Container for simulation time-history data and KPIs."""

    def __init__(self):
        self.time = []
        self.pos = []        # [x, y, z]
        self.vel = []        # [vx, vy, vz]
        self.euler = []      # [roll, pitch, yaw]
        self.omega = []      # [p, q, r]
        self.ctrl = []       # [t1, t2, roll, pitch, yaw]
        self.z_ref = []      # altitude reference
        self.kpis = {}       # Computed after simulation

    def append(self, t, state, ctrl_vals, z_ref_val):
        self.time.append(t)
        self.pos.append(state['pos'].copy())
        self.vel.append(state['vel'].copy())
        self.euler.append(state['euler'].copy())
        self.omega.append(state['omega'].copy())
        self.ctrl.append(np.array(ctrl_vals).copy())
        self.z_ref.append(z_ref_val)

    def to_arrays(self):
        """Convert lists to numpy arrays for analysis."""
        self.time = np.array(self.time)
        self.pos = np.array(self.pos)
        self.vel = np.array(self.vel)
        self.euler = np.array(self.euler)
        self.omega = np.array(self.omega)
        self.ctrl = np.array(self.ctrl)
        self.z_ref = np.array(self.z_ref)

    def compute_kpis(self, settle_window=2.0):
        """
        Compute KPIs from simulation data.

        Parameters
        ----------
        settle_window : float  Time window at end to evaluate steady-state
        """
        self.to_arrays()

        if len(self.time) == 0:
            self.kpis = {'stable': False}
            return self.kpis

        # Altitude tracking error
        z_actual = self.pos[:, 2]
        z_error = self.z_ref - z_actual

        self.kpis['alt_error_rms'] = float(np.sqrt(np.mean(z_error ** 2)))
        self.kpis['alt_error_max'] = float(np.max(np.abs(z_error)))

        # Steady-state analysis (last settle_window seconds)
        t_end = self.time[-1]
        ss_mask = self.time >= (t_end - settle_window)
        if np.sum(ss_mask) > 10:
            z_err_ss = z_error[ss_mask]
            self.kpis['alt_error_ss_rms'] = float(np.sqrt(np.mean(z_err_ss ** 2)))
            self.kpis['alt_error_ss_max'] = float(np.max(np.abs(z_err_ss)))
        else:
            self.kpis['alt_error_ss_rms'] = self.kpis['alt_error_rms']
            self.kpis['alt_error_ss_max'] = self.kpis['alt_error_max']

        # Attitude disturbance
        self.kpis['roll_rms'] = float(np.sqrt(np.mean(self.euler[:, 0] ** 2)))
        self.kpis['pitch_rms'] = float(np.sqrt(np.mean(self.euler[:, 1] ** 2)))

        # Control saturation (fraction of time any ctrl is at ±1)
        ctrl_abs = np.abs(self.ctrl)
        saturated = ctrl_abs >= 0.99
        sat_fraction = np.mean(np.any(saturated, axis=1))
        self.kpis['ctrl_saturation_rate'] = float(sat_fraction)

        # Stability check: altitude didn't diverge
        stable = (np.max(z_actual) < 10.0) and (np.min(z_actual) > -1.0)
        self.kpis['stable'] = bool(stable)

        # Settling time (time to reach within 5% of reference and stay)
        if stable and len(z_error) > 0:
            threshold = 0.05 * Z_REF
            settled = np.abs(z_error) < threshold
            if np.any(settled):
                # Find first time it settles and stays settled
                for i in range(len(settled)):
                    if np.all(settled[i:min(i + 50, len(settled))]):
                        self.kpis['settling_time'] = float(self.time[i])
                        break
                else:
                    self.kpis['settling_time'] = float(t_end)
            else:
                self.kpis['settling_time'] = float(t_end)
        else:
            self.kpis['settling_time'] = float('inf')

        return self.kpis


class MarsSimulator:
    """
    MuJoCo simulation manager for the Mars testbed.

    Handles model loading, density adjustment, simulation execution,
    and result collection.
    """

    def __init__(self, xml_path: str = None, headless: bool = True):
        self.xml_path = xml_path or SCENE_MARS_XML
        self.headless = headless
        self.model, self.data = load_mars_model(self.xml_path)
        self.dt = self.model.opt.timestep

        # Cache IDs
        self.body_id = get_body_id(self.model, "ingenuity")
        self.top_idx, self.bot_idx = get_rotor_joint_indices(self.model)

    def set_density(self, rho: float):
        """Override atmospheric density (for E2 sweep)."""
        self.model.opt.density = rho

    def reset(self, z_init: float = None):
        """Reset simulation state."""
        reset_model(self.model, self.data, z_init or INIT_POS_Z)

    def run_open_loop(self, ctrl_sequence: list,
                      duration: float = None) -> SimResult:
        """
        Run open-loop simulation with time-varying ctrl inputs.

        Parameters
        ----------
        ctrl_sequence : list of (t_start, t_end, ctrl_array)
            Each entry defines ctrl values for a time interval.
        duration : float  Total duration (inferred from sequence if None)

        Returns
        -------
        SimResult with time history
        """
        if duration is None:
            duration = max(t_end for _, t_end, _ in ctrl_sequence)

        self.reset()
        result = SimResult()
        n_steps = int(duration / self.dt)

        for step in range(n_steps):
            t = step * self.dt

            # Find active ctrl command
            ctrl_vals = np.zeros(self.model.nu)
            for t_start, t_end, ctrl in ctrl_sequence:
                if t_start <= t < t_end:
                    ctrl_vals[:len(ctrl)] = ctrl
                    break

            self.data.ctrl[:] = ctrl_vals[:self.model.nu]
            mujoco.mj_step(self.model, self.data)

            state = extract_state(self.data)
            result.append(t, state, self.data.ctrl[:self.model.nu].copy(), 0.0)

        result.to_arrays()
        return result

    def run_hover(self, ctrl_thrust: float, ctrl_yaw: float = 0.0,
                  z_ref: float = Z_REF, duration: float = SIM_DURATION,
                  use_controller: bool = True,
                  viewer: bool = None) -> SimResult:
        """
        Run hover simulation with PID controller.

        Parameters
        ----------
        ctrl_thrust : float  Feedforward thrust ctrl (from BEMT)
        ctrl_yaw    : float  Feedforward yaw ctrl (from BEMT)
        z_ref       : float  Target altitude (m)
        duration    : float  Simulation duration (s)
        use_controller : bool  If True, use PID; if False, open-loop
        viewer      : bool  Override headless setting

        Returns
        -------
        SimResult with KPIs computed
        """
        show_viewer = viewer if viewer is not None else (not self.headless)

        self.reset()
        result = SimResult()

        # Initialize controllers
        alt_pid = AltitudePID(ff_thrust=ctrl_thrust)
        att_pd = AttitudePD()

        n_steps = int(duration / self.dt)

        def _run_sim(viewer_ctx=None):
            for step in range(n_steps):
                t = step * self.dt
                state = extract_state(self.data)

                # Sensor data
                try:
                    laser, gyro, acc, quat = get_sensor_data(self.model, self.data)
                    z_meas = laser if laser > 0 else state['pos'][2]
                except Exception:
                    z_meas = state['pos'][2]
                    gyro = state['omega']

                roll, pitch, yaw = state['euler']
                p, q, r = state['omega']

                if use_controller:
                    # Altitude PID
                    thrust_cmd = alt_pid.compute(z_ref, z_meas, self.dt)

                    # Attitude PD (hold level)
                    x_cmd, y_cmd, z_cmd = att_pd.compute(
                        0.0, 0.0, 0.0,
                        roll, pitch, yaw,
                        p, q, r,
                    )

                    # Apply
                    self.data.ctrl[IDX_THRUST1] = thrust_cmd
                    self.data.ctrl[IDX_THRUST2] = thrust_cmd
                    self.data.ctrl[IDX_ROLL] = x_cmd
                    self.data.ctrl[IDX_PITCH] = y_cmd
                    if self.model.nu > IDX_YAW:
                        self.data.ctrl[IDX_YAW] = ctrl_yaw + z_cmd * 0.1
                else:
                    self.data.ctrl[IDX_THRUST1] = ctrl_thrust
                    self.data.ctrl[IDX_THRUST2] = ctrl_thrust
                    self.data.ctrl[IDX_ROLL] = 0.0
                    self.data.ctrl[IDX_PITCH] = 0.0
                    if self.model.nu > IDX_YAW:
                        self.data.ctrl[IDX_YAW] = ctrl_yaw

                # Rotor visual
                update_rotor_visual(
                    self.data, self.top_idx, self.bot_idx,
                    self.data.ctrl[IDX_THRUST1], self.dt,
                )

                # Physics step
                mujoco.mj_step(self.model, self.data)

                # Record
                ctrl_record = self.data.ctrl[:self.model.nu].copy()
                result.append(t, state, ctrl_record, z_ref)

                # Viewer sync
                if viewer_ctx is not None:
                    update_tracking_camera(viewer_ctx, self.data, self.body_id)
                    viewer_ctx.sync()

                # Early exit if diverged
                if abs(state['pos'][2]) > 20.0:
                    break

        if show_viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as v:
                _run_sim(v)
        else:
            _run_sim(None)

        result.compute_kpis()
        return result

    def run_trajectory(self, trajectory_fn, duration: float = SIM_DURATION,
                       ctrl_thrust: float = None,
                       viewer: bool = None) -> SimResult:
        """
        Run trajectory tracking simulation (for E3).

        Parameters
        ----------
        trajectory_fn : callable(t) -> (x_ref, y_ref, z_ref)
        duration      : float
        ctrl_thrust   : float  Feedforward thrust (default: HOVER_CTRL)

        Returns
        -------
        SimResult
        """
        from controller import ForwardFlightController

        show_viewer = viewer if viewer is not None else (not self.headless)

        if ctrl_thrust is None:
            ctrl_thrust = HOVER_CTRL

        self.reset()
        result = SimResult()

        alt_pid = AltitudePID(ff_thrust=ctrl_thrust)
        att_pd = AttitudePD()
        fwd_ctrl = ForwardFlightController()

        n_steps = int(duration / self.dt)

        def _run_sim(viewer_ctx=None):
            for step in range(n_steps):
                t = step * self.dt
                state = extract_state(self.data)
                pos = state['pos']
                vel = state['vel']
                roll, pitch, yaw = state['euler']
                p, q, r = state['omega']

                # Get reference
                ref = trajectory_fn(t)
                x_ref, y_ref, z_ref = ref[0], ref[1], ref[2]

                # Outer loop: position -> tilt reference
                roll_ref, pitch_ref = fwd_ctrl.compute(
                    np.array([x_ref, y_ref, z_ref]), pos, vel
                )

                # Altitude PID
                z_meas = pos[2]
                thrust_cmd = alt_pid.compute(z_ref, z_meas, self.dt)

                # Tilt compensation
                tilt_mag = np.sqrt(roll ** 2 + pitch ** 2)
                cos_comp = 1.0 / max(np.cos(tilt_mag), 0.85)
                thrust_cmd = min(thrust_cmd * cos_comp, 1.0)

                # Attitude PD
                x_cmd, y_cmd, z_cmd = att_pd.compute(
                    roll_ref, pitch_ref, 0.0,
                    roll, pitch, yaw,
                    p, q, r,
                )

                self.data.ctrl[IDX_THRUST1] = thrust_cmd
                self.data.ctrl[IDX_THRUST2] = thrust_cmd
                self.data.ctrl[IDX_ROLL] = x_cmd
                self.data.ctrl[IDX_PITCH] = y_cmd
                if self.model.nu > IDX_YAW:
                    self.data.ctrl[IDX_YAW] = z_cmd * 0.1

                update_rotor_visual(
                    self.data, self.top_idx, self.bot_idx,
                    thrust_cmd, self.dt,
                )
                mujoco.mj_step(self.model, self.data)

                ctrl_record = self.data.ctrl[:self.model.nu].copy()
                result.append(t, state, ctrl_record, z_ref)

                if viewer_ctx is not None:
                    update_tracking_camera(viewer_ctx, self.data, self.body_id)
                    viewer_ctx.sync()

                if abs(pos[2]) > 20.0:
                    break

        if show_viewer:
            with mujoco.viewer.launch_passive(self.model, self.data) as v:
                _run_sim(v)
        else:
            _run_sim(None)

        result.compute_kpis()
        return result
