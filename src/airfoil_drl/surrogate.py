"""Stage 3 — the AirfRANS lift/drag surrogate.

Moved from ``notebooks/03_airfrans_cfd_surrogate.ipynb`` (case-name parsing of
Part A, the Keras MLP of Section 14, the ``score`` helper of the experiments)
and from ``notebooks/04_airfoil_gust_control_cfd_surrogate.ipynb`` (the NumPy
re-implementation of the trained network, Section 4).  TensorFlow is imported
only inside the functions that need it.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Columns of data/airfrans_4digit_coefficients.csv used as inputs and outputs.
FEATURES = ["Re_M", "alpha_deg", "m", "p", "t"]
TARGETS = ["CL", "CD"]

# Kinematic viscosity of the AirfRANS simulations, used to turn the inlet velocity into a Reynolds number.
NU_DATASET = 1.56e-5


def geometry_parameters_from_name(name):
    """NACA parameters encoded in an AirfRANS case name (``airFoil2D_SST_<U>_<alpha>_<m>_<p>_<t>``)."""
    parts = name.split("_")
    return [float(value) for value in parts[4:]]


def is_four_digit_case(name):
    return len(geometry_parameters_from_name(name)) == 3


def build_surrogate_model(n_features=len(FEATURES), n_targets=len(TARGETS), layers=(64, 64, 32), learning_rate=1e-3):
    """The compiled tanh MLP of Section 14 (64-64-32 by default, 6,690 parameters for 5 inputs and 2 outputs)."""
    import tensorflow as tf

    net = tf.keras.Sequential([tf.keras.layers.Input(shape=(n_features,))] +
                              [tf.keras.layers.Dense(u, activation="tanh") for u in layers] +
                              [tf.keras.layers.Dense(n_targets)])
    net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return net


def score(predict, test_frame, targets=TARGETS):
    """RMSE, MAE and R2 of ``predict(frame) -> (n, len(targets))`` on a test DataFrame."""
    pred = predict(test_frame); out = {}
    for j, name in enumerate(targets):
        yt = test_frame[name].to_numpy()
        out[f"{name}_RMSE"] = float(np.sqrt(mean_squared_error(yt, pred[:, j])))
        out[f"{name}_MAE"] = float(mean_absolute_error(yt, pred[:, j]))
        out[f"{name}_R2"] = float(r2_score(yt, pred[:, j]))
    return out


def dense_layers(keras_model):
    """Weights, biases and activation names of a Sequential MLP, in forward order."""
    return [(layer.get_weights()[0], layer.get_weights()[1], layer.activation.__name__)
            for layer in keras_model.layers]


def mlp_forward(x, layers):
    """Evaluate a list of (W, b, activation) dense layers on ``x`` with NumPy (tanh or linear only)."""
    h = np.asarray(x, dtype=float)
    for w, b, act in layers:
        h = h @ w + b
        if act == "tanh":
            h = np.tanh(h)
        elif act != "linear":
            raise ValueError(act)
    return h


def make_numpy_surrogate(keras_model, x_scaler, y_scaler):
    """Return ``f(reynolds, alpha_deg, m, p, t) -> (cl, cd)``: the trained network evaluated with NumPy.

    Identical to the Keras prediction (max difference about 3e-7 in the notebook) and
    roughly a hundred times faster per call, which is what makes training against the
    surrogate practical.
    """
    _layers = dense_layers(keras_model)
    _xm, _xs = x_scaler.mean_, x_scaler.scale_
    _ym, _ys = y_scaler.mean_, y_scaler.scale_

    def fast_predict_airfrans_aero(reynolds, alpha_deg, m=2.0, p=4.0, t=12.0):
        h = (np.array([reynolds / 1.0e6, alpha_deg, m, p, t]) - _xm) / _xs
        h = mlp_forward(h, _layers)
        cl, cd = h * _ys + _ym
        return float(cl), float(cd)

    return fast_predict_airfrans_aero


def load_numpy_surrogate(model_file, scaler_file):
    """Load a ``.keras`` surrogate and its ``.joblib`` scalers and return the NumPy predictor."""
    import joblib
    import tensorflow as tf

    net = tf.keras.models.load_model(model_file)
    sc = joblib.load(scaler_file)
    return make_numpy_surrogate(net, sc["x_scaler"], sc["y_scaler"])
