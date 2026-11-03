"""The saved Stable-Baselines3 agents, loaded on CPU and flown through the extracted environments.

These are the headline numbers of the README; the tolerances allow for a different
torch/BLAS build than the one the notebooks ran on.
"""
import numpy as np
import pytest
from stable_baselines3 import PPO, SAC, TD3

from airfoil_drl.ks import KSSystem
from airfoil_drl.metrics import episode_metrics
from airfoil_drl.surrogate_env import AirfoilSurrogateGymEnv
from airfoil_drl.thin_airfoil import AirfoilGymEnv, run_episode, suite_metrics


def test_stage2_td3_fixed_gust_matches_notebook(repo_root):
    agent = TD3.load(repo_root / "models/02_airfoil_gust_control_thin_airfoil/airfoil_0d_td3_final.zip", device="cpu")
    m = episode_metrics(run_episode(AirfoilGymEnv(lift_model="linear", random_gust=False), agent, seed=123))
    assert m["CL_RMSE"] == pytest.approx(0.01663, abs=2e-4)           # notebook Section 11: 0.01663
    assert m["max_abs_flap_deg"] == pytest.approx(7.96, abs=0.05)
    row = suite_metrics(agent, "TD3")
    assert row["mean_RMSE"] == pytest.approx(0.016543, abs=2e-4)       # q1_algorithm_comparison.csv


def test_stage4_sac_fixed_gust_matches_notebook(repo_root, numpy_surrogate):
    agent = SAC.load(repo_root / "models/04_airfoil_gust_control_cfd_surrogate/airfoil_surrogate_sac_final.zip", device="cpu")
    env = AirfoilSurrogateGymEnv(numpy_surrogate, random_gust=False)
    obs, _ = env.reset(seed=123)
    data = {k: [] for k in ("cl", "cl_target", "cd", "flap_deg", "reward")}
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action); done = term or trunc
        for k in data:
            data[k].append(r if k == "reward" else info[k])
    m = episode_metrics({k: np.asarray(v) for k, v in data.items()})
    assert m["CL_RMSE"] == pytest.approx(0.059620, abs=5e-4)           # notebook Section 11: 0.059620
    assert m["max_abs_flap_deg"] == pytest.approx(9.61, abs=0.1)


def test_stage1_ppo_beats_no_control_on_unseen_chaotic_states(repo_root):
    """Notebook Section 10: no control 114.3 +/- 3.9, PPO 14.63 +/- 2.54 over 20 states.

    The chaotic attractor amplifies FFT round-off between platforms, so only the
    order of magnitude of the two costs is checked, not the exact digits.
    """
    sys22 = KSSystem()
    pool = sys22.attractor()                      # same seed and transient as the notebook
    training_pool, test_pool = pool[:400], pool[500:]
    scale = max(float(np.std(training_pool)), 0.1)
    agent = PPO.load(repo_root / "models/01_ks_control_gymnasium/ppo/ks_ppo_final.zip", device="cpu")

    def cost(initial, controller, horizon=80, effort_weight=0.01):
        state, total = np.asarray(initial, np.float32).copy(), 0.0
        for _ in range(horizon):
            a = np.clip(controller(state), -1.0, 1.0)
            state = sys22.step(state, a)
            total += np.mean(state**2) + effort_weight * np.sum(a**2)
        return total

    def ppo(state):
        obs = np.clip(np.asarray(state, np.float32) / scale, -10.0, 10.0)
        return np.asarray(agent.predict(obs, deterministic=True)[0], np.float32)

    initials = test_pool[::5]
    free = np.mean([cost(u0, lambda s: np.zeros(5, np.float32)) for u0 in initials])
    ctrl = np.mean([cost(u0, ppo) for u0 in initials])
    assert 90.0 < free < 140.0
    assert ctrl < 25.0
