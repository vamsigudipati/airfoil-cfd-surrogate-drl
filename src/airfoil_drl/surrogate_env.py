"""Stage 4 — the gust-control environment driven by the CFD surrogate.

Moved from ``notebooks/04_airfoil_gust_control_cfd_surrogate.ipynb``.  The
notebook's ``AirfoilSurrogateGymEnv`` looked up the module-level
``predict_airfrans_aero``; here the aerodynamic law is passed explicitly as
``aero_fn(reynolds, alpha_deg, m, p, t) -> (cl, cd)`` and the notebook binds it
in a two-line subclass, so the behaviour is unchanged.
"""

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .thin_airfoil import one_minus_cosine_gust


def low_order_lift(alpha_rad):
    return 2.0 * np.pi * alpha_rad


def low_order_aero(
    alpha_eff_deg,
    cl_current,
    flap_deg,
    cd0=0.012,
    induced_drag_factor=0.025,
    flap_drag_factor=0.08,
):
    """Thin-airfoil lift and the Stage 2 drag model, used by the model-fidelity experiment."""
    alpha_eff_rad = np.deg2rad(alpha_eff_deg)
    cl_qs = low_order_lift(alpha_eff_rad)

    flap_rad = np.deg2rad(flap_deg)

    cd = (
        cd0
        + induced_drag_factor * cl_current**2
        + flap_drag_factor * flap_rad**2
    )

    return float(cl_qs), float(cd)


class AirfoilSurrogateGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        aero_fn,
        dt=0.05,
        horizon=120,
        U_inf=30.0,
        alpha0_deg=5.0,
        cl_target=None,
        max_flap_deg=20.0,
        flap_effectiveness=0.55,
        tau_lift=0.18,
        tau_flap=0.10,
        reynolds=4.0e6,
        naca_m=2.0,
        naca_p=4.0,
        naca_t=12.0,
        random_gust=True,
        gust_amplitude_range=(-3.0, 3.0),
        gust_duration_range=(0.8, 1.8),
        gust_start_range=(1.0, 3.0),
        effort_weight=0.015,
        smoothness_weight=0.020,
        drag_weight=0.020,
    ):
        super().__init__()

        self.aero_fn = aero_fn

        self.dt = dt
        self.horizon = horizon
        self.U_inf = U_inf
        self.alpha0_deg = alpha0_deg
        self.max_flap_deg = max_flap_deg
        self.flap_effectiveness = flap_effectiveness
        self.tau_lift = tau_lift
        self.tau_flap = tau_flap

        self.reynolds = reynolds
        self.naca_m = naca_m
        self.naca_p = naca_p
        self.naca_t = naca_t

        self.random_gust = random_gust
        self.gust_amplitude_range = gust_amplitude_range
        self.gust_duration_range = gust_duration_range
        self.gust_start_range = gust_start_range

        self.effort_weight = effort_weight
        self.smoothness_weight = smoothness_weight
        self.drag_weight = drag_weight

        # If no target is prescribed, use the zero-gust equilibrium lift.
        cl_equilibrium, _ = self.aero_fn(
            self.reynolds,
            self.alpha0_deg,
            self.naca_m,
            self.naca_p,
            self.naca_t,
        )
        self.cl_target = float(cl_equilibrium if cl_target is None else cl_target)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, -2.0, -1.5], dtype=np.float32),
            high=np.array([ 2.0,  2.0,  2.0,  1.5], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def _gust_velocity(self, t):
        return one_minus_cosine_gust(
            t,
            self.gust_start,
            self.gust_duration,
            self.gust_amplitude,
        )

    def _gust_angle_deg(self, t):
        w_g = self._gust_velocity(t)
        return np.rad2deg(np.arctan2(w_g, self.U_inf))

    def _equivalent_alpha_deg(self, gust_angle_deg):
        return (
            self.alpha0_deg
            + gust_angle_deg
            + self.flap_effectiveness * self.flap_deg
        )

    def _aero(self, gust_angle_deg):
        alpha_eq_deg = self._equivalent_alpha_deg(gust_angle_deg)

        cl_qs, cd = self.aero_fn(
            self.reynolds,
            alpha_eq_deg,
            self.naca_m,
            self.naca_p,
            self.naca_t,
        )

        return cl_qs, cd, alpha_eq_deg

    def _get_obs(self):
        gust_angle_deg = self._gust_angle_deg(self.t)

        obs = np.array([
            self.cl / 1.5,
            self.cl_target / 1.5,
            gust_angle_deg / 10.0,
            self.flap_deg / self.max_flap_deg,
        ], dtype=np.float32)

        return np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.t = 0.0
        self.prev_action = 0.0
        self.flap_deg = 0.0

        if self.random_gust:
            self.gust_amplitude = self.np_random.uniform(
                *self.gust_amplitude_range
            )
            self.gust_duration = self.np_random.uniform(
                *self.gust_duration_range
            )
            self.gust_start = self.np_random.uniform(
                *self.gust_start_range
            )
        else:
            self.gust_amplitude = 2.5
            self.gust_duration = 1.3
            self.gust_start = 2.0

        self.cl, self.cd = self.aero_fn(
            self.reynolds,
            self.alpha0_deg,
            self.naca_m,
            self.naca_p,
            self.naca_t,
        )

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_value = float(np.clip(action[0], -1.0, 1.0))

        flap_command_deg = action_value * self.max_flap_deg

        # First-order actuator dynamics.
        self.flap_deg += (
            self.dt / self.tau_flap
        ) * (flap_command_deg - self.flap_deg)

        gust_angle_deg = self._gust_angle_deg(self.t)

        cl_qs, cd, alpha_eq_deg = self._aero(gust_angle_deg)

        # First-order lift dynamics.
        self.cl += (
            self.dt / self.tau_lift
        ) * (cl_qs - self.cl)

        self.cd = cd

        tracking_error = self.cl - self.cl_target
        delta_action = action_value - self.prev_action

        reward = (
            -tracking_error**2
            -self.effort_weight * action_value**2
            -self.smoothness_weight * delta_action**2
            -self.drag_weight * cd
        )

        self.prev_action = action_value
        self.step_count += 1
        self.t = self.step_count * self.dt

        terminated = False
        truncated = self.step_count >= self.horizon

        info = {
            "time": self.t,
            "cl": self.cl,
            "cl_qs": cl_qs,
            "cl_target": self.cl_target,
            "cd": cd,
            "flap_deg": self.flap_deg,
            "flap_command_deg": flap_command_deg,
            "gust_velocity": self._gust_velocity(self.t),
            "gust_angle_deg": self._gust_angle_deg(self.t),
            "alpha_eq_deg": alpha_eq_deg,
            "tracking_error": tracking_error,
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )
