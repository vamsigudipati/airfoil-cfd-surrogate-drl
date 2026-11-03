"""NACA four-digit outline and a plane rotation, used by the flight animations of Stages 2 and 4."""

import numpy as np


def naca4_outline(m=0.0, p=0.0, t=12.0, n=80):
    """Closed outline of a NACA four-digit section (m, p in digits as in the dataset, t in % chord)."""
    m, p, t = m / 100.0, p / 10.0, t / 100.0
    beta = np.linspace(0.0, np.pi, n); xc = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    if m > 0 and p > 0:
        yc = np.where(xc < p, m / p**2 * (2 * p * xc - xc**2), m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xc - xc**2))
    else:
        yc = np.zeros_like(xc)
    xu, yu, xl, yl = xc, yc + yt, xc, yc - yt
    return np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])


def rotate(xs, ys, angle_deg, cx=0.0, cy=0.0):
    """Rotate points counter-clockwise by ``angle_deg`` about (cx, cy)."""
    a = np.deg2rad(angle_deg)
    xr, yr = xs - cx, ys - cy
    return cx + xr * np.cos(a) - yr * np.sin(a), cy + xr * np.sin(a) + yr * np.cos(a)
