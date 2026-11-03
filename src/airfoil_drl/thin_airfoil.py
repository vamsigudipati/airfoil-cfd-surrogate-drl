"""Stage 2 — gust rejection on a thin-airfoil model.

Moved verbatim from ``notebooks/02_airfoil_gust_control_thin_airfoil.ipynb``:
the one-minus-cosine gust (Section 4), the static lift law (Section 5), the
Gymnasium environment (Section 6), and the rollout / evaluation helpers.
"""

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .metrics import episode_metrics


def one_minus_cosine_gust(t, t0, duration, amplitude):
    if t < t0 or t > t0 + duration:
        return 0.0

    xi = (t - t0) / duration
    return 0.5 * amplitude * (1.0 - np.cos(2.0 * np.pi * xi))


def static_lift_coefficient(alpha_rad, model="linear"):
    if model == "linear":
        return 2.0 * np.pi * alpha_rad

    if model == "nonlinear":
        # Linear behavior at small alpha with smooth saturation at large alpha.
        cl_linear = 2.0 * np.pi * alpha_rad
        cl_max = 1.35
        return cl_max * np.tanh(cl_linear / cl_max)

    raise ValueError("model must be 'linear' or 'nonlinear'")


class AirfoilGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=0.05,
        horizon=120,
        U_inf=30.0,
        alpha0_deg=5.0,
        cl_target=0.55,
        max_flap_deg=20.0,
        flap_effectiveness=0.55,
        tau_lift=0.18,
        tau_flap=0.10,
        cd0=0.012,
        induced_drag_factor=0.025,
        flap_drag_factor=0.08,
        lift_model="linear",
        random_gust=True,
        gust_amplitude_range=(-3.0, 3.0),
        gust_duration_range=(0.8, 1.8),
        gust_start_range=(1.0, 3.0),
        effort_weight=0.015,
        smoothness_weight=0.020,
        drag_weight=0.020,
    ):
        super().__init__()

        self.dt = dt
        self.horizon = horizon
        self.U_inf = U_inf
        self.alpha0_deg = alpha0_deg
        self.cl_target = cl_target
        self.max_flap_deg = max_flap_deg
        self.flap_effectiveness = flap_effectiveness
        self.tau_lift = tau_lift
        self.tau_flap = tau_flap

        self.cd0 = cd0
        self.induced_drag_factor = induced_drag_factor
        self.flap_drag_factor = flap_drag_factor
        self.lift_model = lift_model

        self.random_gust = random_gust
        self.gust_amplitude_range = gust_amplitude_range
        self.gust_duration_range = gust_duration_range
        self.gust_start_range = gust_start_range

        self.effort_weight = effort_weight
        self.smoothness_weight = smoothness_weight
        self.drag_weight = drag_weight

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # The observations are scaled and clipped.
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

    def _aero(self, gust_angle_deg):
        alpha_eff_deg = (
            self.alpha0_deg
            + gust_angle_deg
            + self.flap_effectiveness * self.flap_deg
        )
        alpha_eff_rad = np.deg2rad(alpha_eff_deg)

        cl_qs = static_lift_coefficient(alpha_eff_rad, self.lift_model)

        flap_rad = np.deg2rad(self.flap_deg)
        cd = (
            self.cd0
            + self.induced_drag_factor * self.cl**2
            + self.flap_drag_factor * flap_rad**2
        )

        return cl_qs, cd, alpha_eff_deg

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
            self.gust_amplitude = self.np_random.uniform(*self.gust_amplitude_range)
            self.gust_duration = self.np_random.uniform(*self.gust_duration_range)
            self.gust_start = self.np_random.uniform(*self.gust_start_range)
        else:
            self.gust_amplitude = 2.5
            self.gust_duration = 1.3
            self.gust_start = 2.0

        # Initialize the lift close to its zero-gust equilibrium.
        alpha_initial = np.deg2rad(self.alpha0_deg)
        self.cl = float(static_lift_coefficient(alpha_initial, self.lift_model))

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_value = float(np.clip(action[0], -1.0, 1.0))

        flap_command_deg = action_value * self.max_flap_deg

        # First-order flap dynamics.
        self.flap_deg += (
            self.dt / self.tau_flap
        ) * (flap_command_deg - self.flap_deg)

        gust_angle_deg = self._gust_angle_deg(self.t)
        cl_qs, cd, alpha_eff_deg = self._aero(gust_angle_deg)

        # First-order unsteady lift dynamics.
        self.cl += (
            self.dt / self.tau_lift
        ) * (cl_qs - self.cl)

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
            "cl_target": self.cl_target,
            "cd": cd,
            "flap_deg": self.flap_deg,
            "flap_command_deg": flap_command_deg,
            "gust_velocity": self._gust_velocity(self.t),
            "gust_angle_deg": self._gust_angle_deg(self.t),
            "alpha_eff_deg": alpha_eff_deg,
            "tracking_error": tracking_error,
        }

        return self._get_obs(), float(reward), terminated, truncated, info


class ZeroPolicy:
    """The uncontrolled airfoil: a policy that always commands a zero flap."""

    def predict(self, obs, deterministic=True):
        return np.array([0.0], dtype=np.float32), None


def run_episode(env, model=None, seed=0):
    """One episode; ``model=None`` samples random actions (Section 7 sanity check)."""
    obs, _ = env.reset(seed=seed)

    data = {
        "time": [],
        "cl": [],
        "cl_target": [],
        "cd": [],
        "flap_deg": [],
        "gust_velocity": [],
        "gust_angle_deg": [],
        "reward": [],
    }

    terminated = truncated = False

    while not (terminated or truncated):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        data["time"].append(info["time"])
        data["cl"].append(info["cl"])
        data["cl_target"].append(info["cl_target"])
        data["cd"].append(info["cd"])
        data["flap_deg"].append(info["flap_deg"])
        data["gust_velocity"].append(info["gust_velocity"])
        data["gust_angle_deg"].append(info["gust_angle_deg"])
        data["reward"].append(reward)

    return {key: np.asarray(value) for key, value in data.items()}


def evaluate_on_gust(
    model,
    amplitude,
    duration,
    start=2.0,
    seed=0,
):
    """Fly one prescribed gust with the linear lift law (Section 12)."""
    env = AirfoilGymEnv(
        lift_model="linear",
        random_gust=False,
    )

    obs, _ = env.reset(seed=seed)
    env.gust_amplitude = amplitude
    env.gust_duration = duration
    env.gust_start = start

    data = {
        "time": [],
        "cl": [],
        "cl_target": [],
        "cd": [],
        "flap_deg": [],
        "gust_velocity": [],
        "gust_angle_deg": [],
        "reward": [],
    }

    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        data["time"].append(info["time"])
        data["cl"].append(info["cl"])
        data["cl_target"].append(info["cl_target"])
        data["cd"].append(info["cd"])
        data["flap_deg"].append(info["flap_deg"])
        data["gust_velocity"].append(info["gust_velocity"])
        data["gust_angle_deg"].append(info["gust_angle_deg"])
        data["reward"].append(reward)

    return {key: np.asarray(value) for key, value in data.items()}


# Gusts (W0 [m/s], Tg [s]) of Section 12 and the evaluation suite of the experiments.
TEST_CASES = [
    (-3.5, 1.0),
    (-2.0, 1.7),
    ( 2.0, 1.1),
    ( 3.5, 1.6),
]
SUITE = [(2.5, 1.3)] + list(TEST_CASES)          # (W0 [m/s], Tg [s])


def fly(policy, amplitude, duration, start=2.0, env_cls=AirfoilGymEnv, **env_kw):
    """Fly one prescribed gust and record the trace, including the commanded action."""
    env = env_cls(random_gust=False, **env_kw)
    obs, _ = env.reset(seed=0)
    env.gust_amplitude, env.gust_duration, env.gust_start = amplitude, duration, start
    obs = env._get_obs()
    keys = ["time", "cl", "cl_target", "cd", "flap_deg", "gust_angle_deg", "alpha_eff_deg"]
    out = {k: [] for k in keys + ["reward", "action"]}
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        for k in keys:
            out[k].append(info[k])
        out["reward"].append(reward); out["action"].append(float(np.clip(np.ravel(action)[0], -1, 1)))
    return {k: np.asarray(v) for k, v in out.items()}


def suite_metrics(policy, label, gusts=SUITE, **fly_kw):
    """One row of the experiment tables: lift RMSE per gust and aggregates over the suite."""
    row = {"configuration": label}
    rmses, flaps, mflaps, cds, rough = [], [], [], [], []
    for W0, Tg in gusts:
        tr = fly(policy, W0, Tg, **fly_kw)
        m = episode_metrics(tr)
        rmses.append(m["CL_RMSE"]); flaps.append(m["max_abs_flap_deg"]); mflaps.append(m["mean_abs_flap_deg"])
        cds.append(m["mean_CD"]); rough.append(np.mean(np.abs(np.diff(tr["action"]))))
        row[f"RMSE W0={W0:+.1f},Tg={Tg:.1f}"] = m["CL_RMSE"]
    row.update({"mean_RMSE": np.mean(rmses), "max_abs_flap_deg": np.max(flaps), "mean_abs_flap_deg": np.mean(mflaps),
                "mean_CD": np.mean(cds), "action_roughness": np.mean(rough)})
    return row
