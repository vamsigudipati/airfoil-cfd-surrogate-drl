"""Stage 2 helpers: gust, lift law, AirfoilGymEnv and the evaluation helpers."""
import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from airfoil_drl.metrics import episode_metrics
from airfoil_drl.thin_airfoil import (SUITE, TEST_CASES, AirfoilGymEnv, ZeroPolicy, evaluate_on_gust, fly,
                                      one_minus_cosine_gust, run_episode, static_lift_coefficient, suite_metrics)


def test_gust_is_zero_outside_and_peaks_at_the_middle():
    assert one_minus_cosine_gust(0.5, 2.0, 1.0, 3.0) == 0.0
    assert one_minus_cosine_gust(3.5, 2.0, 1.0, 3.0) == 0.0
    assert one_minus_cosine_gust(2.5, 2.0, 1.0, 3.0) == pytest.approx(3.0)
    assert one_minus_cosine_gust(2.0, 2.0, 1.0, 3.0) == pytest.approx(0.0)
    assert one_minus_cosine_gust(2.25, 2.0, 1.0, 3.0) == pytest.approx(one_minus_cosine_gust(2.75, 2.0, 1.0, 3.0))


def test_static_lift_models():
    alpha = np.deg2rad(np.array([-5.0, 0.0, 2.0, 5.0]))
    np.testing.assert_allclose(static_lift_coefficient(alpha), 2 * np.pi * alpha)
    small = np.deg2rad(np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_allclose(static_lift_coefficient(small, "nonlinear"), 2 * np.pi * small, atol=1e-3)  # linear at small alpha
    assert abs(static_lift_coefficient(alpha[-1], "nonlinear")) < abs(2 * np.pi * alpha[-1])             # already bending at 5 deg
    assert abs(static_lift_coefficient(np.deg2rad(60.0), "nonlinear")) < 1.35  # saturates below cl_max
    with pytest.raises(ValueError):
        static_lift_coefficient(0.1, "quadratic")


def test_env_passes_gymnasium_check_and_has_the_documented_spaces():
    env = AirfoilGymEnv()
    check_env(env, warn=False)
    assert env.action_space.shape == (1,) and env.observation_space.shape == (4,)
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)


def test_fixed_gust_episode_is_deterministic_and_120_steps_long():
    env = AirfoilGymEnv(lift_model="linear", random_gust=False)
    a = run_episode(env, ZeroPolicy(), seed=1)
    b = run_episode(AirfoilGymEnv(lift_model="linear", random_gust=False), ZeroPolicy(), seed=2)
    assert len(a["time"]) == 120 and a["time"][-1] == pytest.approx(6.0)
    np.testing.assert_array_equal(a["cl"], b["cl"])            # the fixed gust ignores the seed
    assert np.all(a["flap_deg"] == 0.0)
    assert a["gust_velocity"].max() == pytest.approx(2.5, abs=0.05)
    assert a["cl"][0] == pytest.approx(2 * np.pi * np.deg2rad(5.0), abs=1e-3)   # starts at equilibrium


def test_uncontrolled_metrics_reproduce_the_notebook_output():
    # Stage 2 notebook, Section 11 output: Uncontrolled CL_RMSE 0.13852, mean_CD 0.02155, return -2.35426
    m = episode_metrics(run_episode(AirfoilGymEnv(lift_model="linear", random_gust=False), ZeroPolicy(), seed=123))
    assert m["CL_RMSE"] == pytest.approx(0.13852, abs=1e-5)
    assert m["mean_CD"] == pytest.approx(0.02155, abs=1e-5)
    assert m["episode_return"] == pytest.approx(-2.35426, abs=1e-5)
    assert m["max_abs_flap_deg"] == 0.0


def test_suite_metrics_reproduce_the_results_csv(repo_root):
    table = pd.read_csv(repo_root / "results/02_airfoil_gust_control_thin_airfoil/q1_algorithm_comparison.csv")
    expected = table[table.configuration == "uncontrolled"].iloc[0]
    row = suite_metrics(ZeroPolicy(), "uncontrolled")
    assert row["mean_RMSE"] == pytest.approx(expected["mean_RMSE"], abs=1e-9)
    assert row["mean_CD"] == pytest.approx(expected["mean_CD"], abs=1e-9)
    for W0, Tg in SUITE:
        key = f"RMSE W0={W0:+.1f},Tg={Tg:.1f}"
        assert row[key] == pytest.approx(expected[key], abs=1e-9)
    assert SUITE[0] == (2.5, 1.3) and SUITE[1:] == TEST_CASES


def test_evaluate_on_gust_and_fly_agree_for_the_zero_policy():
    a = evaluate_on_gust(ZeroPolicy(), amplitude=-3.5, duration=1.0)
    b = fly(ZeroPolicy(), -3.5, 1.0)
    np.testing.assert_allclose(a["cl"], b["cl"])
    assert np.all(b["action"] == 0.0) and "alpha_eff_deg" in b


def test_random_gust_draws_inside_the_configured_ranges():
    env = AirfoilGymEnv(random_gust=True, gust_amplitude_range=(-1.0, 1.0), gust_duration_range=(0.9, 1.0),
                        gust_start_range=(1.0, 1.1))
    for seed in range(5):
        env.reset(seed=seed)
        assert -1.0 <= env.gust_amplitude <= 1.0 and 0.9 <= env.gust_duration <= 1.0 and 1.0 <= env.gust_start <= 1.1
