"""Stage 1 helpers: ETDRK4 coefficients, actuators, KSSystem, LQR, Lyapunov, MPC."""
import numpy as np
import pandas as pd
import pytest

from airfoil_drl.ks import (KSSystem, design_lqr, etdrk4_coefficients, lyapunov_slope, make_mpc,
                            periodic_distance, reduced_linear_model, summarise, twin_divergence)


def test_etdrk4_coefficients_shapes_and_zero_mode_limits():
    L = np.array([0.0, -1.0, 0.75, -16.0])
    dt = 0.25
    E, E2, Q, f1, f2, f3 = etdrk4_coefficients(L, dt)
    for c in (E, E2, Q, f1, f2, f3):
        assert c.shape == L.shape and np.all(np.isfinite(c))
    np.testing.assert_allclose(E, np.exp(dt * L))
    np.testing.assert_allclose(E2, np.exp(0.5 * dt * L))
    # For a zero eigenvalue the phi-functions reduce to dt/2, dt/6, dt/6, dt/6.
    np.testing.assert_allclose([Q[0], f1[0], f2[0], f3[0]], [dt / 2, dt / 6, dt / 6, dt / 6], rtol=1e-10)


def test_periodic_distance_wraps_and_is_bounded():
    x = np.linspace(0.0, 22.0, 64, endpoint=False)
    d = periodic_distance(x, 21.0, 22.0)
    assert np.all(np.abs(d) <= 11.0)
    assert np.isclose(d[0], 1.0)          # x=0 is one unit past the centre, going round the domain
    assert np.isclose(periodic_distance(np.array([1.0]), 1.0, 22.0)[0], 0.0)


def test_kssystem_grid_actuators_and_unstable_pairs_match_results_table(repo_root):
    sys22 = KSSystem()
    assert sys22.n == 64 and sys22.n_action == 5 and sys22.n_sub == 4
    assert sys22.Phi.shape == (64, 5)
    np.testing.assert_allclose(sys22.Phi.mean(axis=0), 0.0, atol=1e-12)        # mean-free shapes
    np.testing.assert_allclose(np.abs(sys22.Phi).max(axis=0), 1.0)              # unit amplitude
    table = pd.read_csv(repo_root / "results/01_ks_control_gymnasium/ex1_domain_length.csv")
    for L, pairs in zip(table.L, table.unstable_pairs):
        assert KSSystem(length=L).n_unstable_pairs == pairs


def test_kssystem_step_keeps_zero_mean_and_zero_is_a_fixed_point():
    sys22 = KSSystem()
    rng = np.random.default_rng(0)
    u = rng.normal(size=64).astype(np.float32)
    v = sys22.step(u, np.zeros(5))
    assert v.shape == (64,) and v.dtype == np.float32
    assert abs(float(v.mean())) < 1e-6
    assert sys22.step(np.zeros(64), np.zeros(5), np.float64).max() == 0.0
    # Forcing with one actuator moves the field in the direction of its shape.
    w = sys22.step(np.zeros(64), np.array([1.0, 0, 0, 0, 0]), np.float64)
    assert np.dot(w, sys22.Phi[:, 0]) > 0


def test_attractor_is_chaotic_and_reproducible():
    sys22 = KSSystem()
    a = sys22.attractor(n_transient=20, n_keep=30)
    b = sys22.attractor(n_transient=20, n_keep=30)
    assert a.shape == (30, 64) and np.array_equal(a, b)
    energy = 0.5 * np.mean(a**2, axis=1)
    assert energy.mean() > 0.1


def test_reduced_model_and_lqr_match_notebook_spectral_radius():
    sys22 = KSSystem()
    Ar, Br, basis = reduced_linear_model(sys22)
    assert Ar.shape == (6, 6) and Br.shape == (6, 5) and basis.shape == (64, 6)
    np.testing.assert_allclose(basis.T @ basis, np.eye(6), atol=1e-12)
    assert max(abs(np.linalg.eigvals(Ar))) > 1.0           # the equilibrium is unstable
    controller, radius = design_lqr(sys22)
    assert radius == pytest.approx(0.8143, abs=2e-4)        # notebook Section 7: 0.81433892891358
    a = controller(np.ones(64))
    assert a.shape == (5,) and a.dtype == np.float32 and np.all(np.abs(a) <= 1.0)


def test_lyapunov_slope_recovers_known_rate():
    t = np.arange(100)
    dist = 1e-4 * np.exp(0.05 * t)
    assert lyapunov_slope(dist, dt=1.0, saturation=1.0) == pytest.approx(0.05, rel=1e-6)
    assert np.isnan(lyapunov_slope(np.array([2.0, 3.0, 4.0])))        # saturated at once: no fit


def test_twin_divergence_starts_at_eps_and_grows_without_control():
    sys22 = KSSystem()
    u0 = sys22.attractor(n_transient=20, n_keep=1)[0]
    d = twin_divergence(sys22, u0, None, steps=40, eps=1e-4)
    assert d.shape == (41,) and d[0] == pytest.approx(1e-4, rel=1e-9)
    assert d[-1] > d[0]


def test_summarise_from_known_frame():
    df = pd.DataFrame({"cost": [1.0, 3.0], "effort_cost": [0.1, 0.3], "final_energy": [0.01, 0.03],
                       "success": [True, False], "returned_to_chaos": [False, False]})
    row = summarise(df, "x")
    assert row["configuration"] == "x" and row["mean_cost"] == 2.0 and row["success_rate"] == 0.5
    assert row["returned_to_chaos_rate"] == 0.0 and row["mean_effort_cost"] == pytest.approx(0.2)


def test_mpc_controller_returns_bounded_action_and_resets():
    sys22 = KSSystem()
    mpc = make_mpc(sys22, horizon=2, block=1, maxiter=3)
    u0 = sys22.attractor(n_transient=20, n_keep=1)[0]
    a = mpc(u0)
    assert a.shape == (5,) and a.dtype == np.float32 and np.all(np.abs(a) <= 1.0)
    mpc.reset()
