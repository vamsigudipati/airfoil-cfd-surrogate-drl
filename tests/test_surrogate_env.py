"""Stage 4 environment with an explicit aerodynamic law."""
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from airfoil_drl.metrics import episode_metrics
from airfoil_drl.surrogate_env import AirfoilSurrogateGymEnv, low_order_aero, low_order_lift
from airfoil_drl.thin_airfoil import ZeroPolicy


def thin_airfoil_aero(reynolds, alpha_deg, m, p, t):
    """A cheap stand-in for the surrogate: 2*pi*alpha and the Stage 2 drag model."""
    cl = low_order_lift(np.deg2rad(alpha_deg))
    return float(cl), low_order_aero(alpha_deg, cl, 0.0)[1]


def test_low_order_aero():
    cl, cd = low_order_aero(5.0, 0.5, 10.0)
    assert cl == pytest.approx(2 * np.pi * np.deg2rad(5.0))
    assert cd == pytest.approx(0.012 + 0.025 * 0.25 + 0.08 * np.deg2rad(10.0) ** 2)


def test_env_with_a_cheap_aero_law_passes_check_env_and_targets_the_equilibrium_lift():
    env = AirfoilSurrogateGymEnv(thin_airfoil_aero)
    check_env(env, warn=False)
    assert env.cl_target == pytest.approx(2 * np.pi * np.deg2rad(5.0))
    assert AirfoilSurrogateGymEnv(thin_airfoil_aero, cl_target=0.3).cl_target == 0.3
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,) and obs[0] == pytest.approx(obs[1])            # starts on target
    obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    assert not terminated and not truncated and reward < 0.0
    assert info["flap_deg"] == pytest.approx(0.5 * 20.0 * 0.05 / 0.10)       # first-order flap step
    assert set(info) >= {"cl", "cl_qs", "cd", "alpha_eq_deg", "gust_angle_deg"}


def test_episode_length_and_reynolds_geometry_are_passed_to_the_aero_law():
    calls = []

    def recording(reynolds, alpha_deg, m, p, t):
        calls.append((reynolds, m, p, t))
        return thin_airfoil_aero(reynolds, alpha_deg, m, p, t)

    env = AirfoilSurrogateGymEnv(recording, random_gust=False, reynolds=2.0e6, naca_m=4.0, naca_p=4.0, naca_t=15.0)
    done, n = False, 0
    env.reset(seed=0)
    while not done:
        _, _, term, trunc, _ = env.step(np.zeros(1, np.float32)); done = term or trunc; n += 1
    assert n == 120 and set(calls) == {(2.0e6, 4.0, 4.0, 15.0)}


def test_uncontrolled_episode_with_the_provided_surrogate_reproduces_the_notebook(numpy_surrogate):
    # Stage 4 notebook, Section 11 table: Uncontrolled CL_RMSE 0.135357, mean_CD 0.010051, return -2.222713
    env = AirfoilSurrogateGymEnv(numpy_surrogate, random_gust=False)
    assert env.cl_target == pytest.approx(0.7595, abs=1e-4)
    obs, _ = env.reset(seed=123)
    data = {k: [] for k in ("cl", "cl_target", "cd", "flap_deg", "reward")}
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(np.zeros(1, np.float32)); done = term or trunc
        for k in data:
            data[k].append(r if k == "reward" else info[k])
    m = episode_metrics({k: np.asarray(v) for k, v in data.items()})
    assert m["CL_RMSE"] == pytest.approx(0.135357, abs=1e-5)
    assert m["mean_CD"] == pytest.approx(0.010051, abs=1e-5)
    assert m["episode_return"] == pytest.approx(-2.222713, abs=1e-5)
    assert isinstance(ZeroPolicy().predict(obs)[0], np.ndarray)
