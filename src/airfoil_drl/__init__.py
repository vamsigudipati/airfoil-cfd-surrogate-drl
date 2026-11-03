"""Pure functions and environments extracted from the four notebooks.

The code here is a verbatim move of definitions that the notebooks used to
define inline; the notebooks now import them back.  Nothing was retrained and
no result changed.  Module map:

- ``ks``            Stage 1: ETDRK4 Kuramoto–Sivashinsky solver, actuators, local LQR, MPC, Lyapunov helpers.
- ``thin_airfoil``  Stage 2: one-minus-cosine gust, thin-airfoil lift law, ``AirfoilGymEnv`` and its evaluation helpers.
- ``surrogate``     Stage 3: AirfRANS case-name parsing, the Keras MLP, regression scoring, a NumPy copy of the trained MLP.
- ``surrogate_env`` Stage 4: ``AirfoilSurrogateGymEnv`` driven by any (Re, alpha, m, p, t) -> (CL, CD) callable, low-order aero.
- ``metrics``       episode metrics shared by Stages 2 and 4.
- ``geometry``      NACA four-digit outline and rotation used by the flight animations.
"""

__all__ = ["ks", "thin_airfoil", "surrogate", "surrogate_env", "metrics", "geometry"]
