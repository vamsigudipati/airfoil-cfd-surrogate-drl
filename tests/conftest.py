import os
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # keep TensorFlow quiet
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU only
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def coefficient_table(repo_root):
    import pandas as pd

    return pd.read_csv(repo_root / "data" / "airfrans_4digit_coefficients.csv")


@pytest.fixture(scope="session")
def numpy_surrogate(repo_root):
    """The provided surrogate of data/ (the one Stage 4 trains against), as a NumPy callable."""
    from airfoil_drl.surrogate import load_numpy_surrogate

    return load_numpy_surrogate(repo_root / "data" / "airfrans_cl_cd_surrogate.keras",
                                repo_root / "data" / "airfrans_cl_cd_scalers.joblib")
