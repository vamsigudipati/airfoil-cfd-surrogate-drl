"""Stage 3 helpers: name parsing, the Keras MLP, scoring and the NumPy forward pass."""
import numpy as np
import pandas as pd
import pytest

from airfoil_drl.surrogate import (FEATURES, TARGETS, build_surrogate_model, dense_layers, geometry_parameters_from_name,
                                   is_four_digit_case, make_numpy_surrogate, mlp_forward, score)


def test_case_name_parsing():
    name = "airFoil2D_SST_58.831_-3.563_2.815_4.916_10.078"
    assert geometry_parameters_from_name(name) == [2.815, 4.916, 10.078]
    assert is_four_digit_case(name)
    assert not is_four_digit_case("airFoil2D_SST_58.831_-3.563_1.0_2.0_3.0_4.0")


def test_coefficient_table_has_the_documented_shape(coefficient_table):
    df = coefficient_table
    assert len(df) == 489 and (df.split == "train").sum() == 378 and (df.split == "test").sum() == 111
    assert set(FEATURES + TARGETS) <= set(df.columns)
    assert df.name.map(is_four_digit_case).all()


def test_build_surrogate_model_parameter_count_and_two_sample_forward_pass():
    model = build_surrogate_model()
    assert model.count_params() == 6690                      # Stage 3 model.summary(): 6,690 parameters
    x = np.array([[4.0, 5.0, 2.0, 4.0, 12.0], [3.0, -2.0, 0.0, 0.0, 15.0]], dtype=np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 2) and np.all(np.isfinite(y))
    small = build_surrogate_model(4, 1, layers=(8,))
    assert small.count_params() == 4 * 8 + 8 + 8 + 1 and small.predict(x[:, :4], verbose=0).shape == (2, 1)


def test_numpy_forward_pass_matches_keras_on_a_two_sample_batch():
    model = build_surrogate_model()
    x = np.array([[0.1, -0.2, 0.3, 0.0, 1.0], [1.0, 2.0, -1.0, 0.5, -0.5]])
    np.testing.assert_allclose(mlp_forward(x, dense_layers(model)), model.predict(x, verbose=0), atol=1e-5)
    with pytest.raises(ValueError):
        mlp_forward(x, [(np.eye(5), np.zeros(5), "relu")])


def test_make_numpy_surrogate_undoes_the_scalers():
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    xs = StandardScaler().fit(rng.normal(size=(20, 5)) * [1, 5, 2, 2, 4] + [4, 5, 3, 3, 12])
    ys = StandardScaler().fit(rng.normal(size=(20, 2)) * [0.5, 0.005] + [0.7, 0.012])
    model = build_surrogate_model()
    f = make_numpy_surrogate(model, xs, ys)
    cl, cd = f(4.0e6, 5.0, 2.0, 4.0, 12.0)
    expected = ys.inverse_transform(model.predict(xs.transform([[4.0, 5.0, 2.0, 4.0, 12.0]]), verbose=0))[0]
    assert cl == pytest.approx(expected[0], abs=1e-5) and cd == pytest.approx(expected[1], abs=1e-6)
    assert f(4.0e6, 5.0) == f(4.0e6, 5.0, 2.0, 4.0, 12.0)           # NACA 2412 defaults


def test_provided_surrogate_reproduces_the_results_csv(repo_root, numpy_surrogate, coefficient_table):
    # results/03_airfrans_cfd_surrogate/provided_surrogate_evaluation.csv, row "provided (used by Stage 4)"
    table = pd.read_csv(repo_root / "results/03_airfrans_cfd_surrogate/provided_surrogate_evaluation.csv")
    row = table[table.surrogate.str.startswith("provided")].iloc[0]
    cl, cd = numpy_surrogate(4.0e6, 5.0, 2.0, 4.0, 12.0)
    assert cl == pytest.approx(row["demo_CL (Re 4e6, a 5, 2412)"], abs=1e-4)   # 0.7595 in the Stage 4 output
    assert cd == pytest.approx(row["demo_CD"], abs=1e-5)

    test = coefficient_table[coefficient_table.split == "test"]

    def predict(frame):
        return np.array([numpy_surrogate(1e6 * r.Re_M, r.alpha_deg, r.m, r.p, r.t) for r in frame.itertuples()])

    s = score(predict, test)
    assert s["CL_RMSE"] == pytest.approx(row["CL_RMSE"], abs=1e-5)
    assert s["CD_RMSE"] == pytest.approx(row["CD_RMSE"], abs=1e-6)
    assert s["CL_R2"] == pytest.approx(row["CL_R2"], abs=1e-4)


def test_score_keys_and_perfect_prediction():
    frame = pd.DataFrame({"CL": [0.1, 0.2, 0.4], "CD": [0.01, 0.02, 0.03]})
    s = score(lambda f: f[["CL", "CD"]].to_numpy(), frame)
    assert set(s) == {"CL_RMSE", "CL_MAE", "CL_R2", "CD_RMSE", "CD_MAE", "CD_R2"}
    assert s["CL_RMSE"] == 0.0 and s["CD_R2"] == 1.0
