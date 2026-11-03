import numpy as np
import pytest

from airfoil_drl.geometry import naca4_outline, rotate
from airfoil_drl.metrics import episode_metrics, regression_metrics


def test_episode_metrics_on_a_known_case():
    data = {"cl": np.array([1.0, 2.0, 2.0]), "cl_target": np.array([1.0, 1.0, 1.0]),
            "flap_deg": np.array([-2.0, 4.0, 0.0]), "cd": np.array([0.01, 0.02, 0.03]), "reward": np.array([-1.0, -2.0, 0.5])}
    m = episode_metrics(data)
    assert m["CL_RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert m["mean_abs_flap_deg"] == pytest.approx(2.0) and m["max_abs_flap_deg"] == 4.0
    assert m["mean_CD"] == pytest.approx(0.02) and m["episode_return"] == pytest.approx(-2.5)


def test_regression_metrics_perfect_and_biased():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert regression_metrics(y, y) == {"MAE": 0.0, "RMSE": 0.0, "R2": 1.0}
    m = regression_metrics(y, y + 0.5)
    assert m["MAE"] == pytest.approx(0.5) and m["RMSE"] == pytest.approx(0.5) and m["R2"] == pytest.approx(1 - 0.25 / 1.25)


def test_naca4_outline_is_closed_symmetric_and_has_the_right_thickness():
    xs, ys = naca4_outline(0.0, 0.0, 12.0, n=200)
    assert xs.shape == ys.shape == (399,)
    assert xs.min() == pytest.approx(0.0) and xs.max() == pytest.approx(1.0)
    assert ys.max() == pytest.approx(0.06, abs=1e-3) and ys.min() == pytest.approx(-0.06, abs=1e-3)
    assert xs[0] == pytest.approx(xs[-1]) and ys[0] == pytest.approx(-ys[-1], abs=1e-12)
    xc, yc = naca4_outline(2.0, 4.0, 12.0, n=200)
    assert yc.max() > 0.06 and (yc.max() + yc.min()) > 0.0       # camber lifts the section


def test_rotate_quarter_turn_about_a_point():
    x, y = rotate(np.array([2.0]), np.array([0.0]), 90.0, cx=1.0, cy=0.0)
    assert x[0] == pytest.approx(1.0) and y[0] == pytest.approx(1.0)
