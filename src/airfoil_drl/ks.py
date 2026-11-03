"""Stage 1 — forced Kuramoto–Sivashinsky equation on a periodic grid.

Everything here is moved verbatim from ``notebooks/01_ks_control_gymnasium.ipynb``:
the ETDRK4 coefficients (Section 3), the periodic actuator distance (Section 4),
the parameterised solver ``KSSystem`` and the LQR / MPC / Lyapunov helpers of the
experiments (Sections 14 and 7 of the executed experiments).
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize


# Compute the ETDRK4 coefficients associated with the linear operator.
# These coefficients depend only on the linear KS dynamics and the time step.
def etdrk4_coefficients(linear_operator, dt, contour_points=32):

    # Exact exponential propagation of the linear term
    E = np.exp(dt * linear_operator)
    E2 = np.exp(0.5 * dt * linear_operator)

    # Complex contour points used to evaluate the ETDRK4 phi-functions
    # accurately and avoid numerical cancellation.
    roots = np.exp(1j*np.pi*(np.arange(1, contour_points+1)-0.5)/contour_points )

    LR = dt * linear_operator[:, None] + roots[None, :]

    # ETDRK4 coefficients
    Q = dt * np.real(
        np.mean((np.exp(LR / 2.0) - 1.0) / LR,axis=1,)
    )

    f1 = dt * np.real(
        np.mean((-4.0- LR+ np.exp(LR) * (4.0 - 3.0 * LR + LR**2))/ LR**3,axis=1)
    )

    f2 = dt * np.real(
        np.mean((2.0+ LR+ np.exp(LR) * (-2.0 + LR))/ LR**3,axis=1)
    )

    f3 = dt * np.real(
        np.mean((-4.0- 3.0 * LR- LR**2+ np.exp(LR) * (4.0 - LR)) / LR**3,axis=1)
    )

    return E, E2, Q, f1, f2, f3


def periodic_distance(points, center, period):
    return (points-center+0.5*period) % period-0.5*period


class KSSystem:
    """Same physics and ETDRK4 solver as Sections 2-4, parameterised by domain length and actuator layout."""

    def __init__(self, length=22.0, n_points=64, n_actuators=5, actuator_centers=None,
                 actuator_width=1.0, dt=0.25, control_interval=1.0):
        self.L, self.n = length, n_points
        self.dx = length / n_points
        self.x = np.arange(n_points) * self.dx
        self.k = 2.0 * np.pi * np.fft.fftfreq(n_points, d=self.dx)
        modes = np.fft.fftfreq(n_points) * n_points
        self.mask = np.abs(modes) <= n_points / 3
        self.n_sub = int(round(control_interval / dt))
        self.E, self.E2, self.Q, self.f1, self.f2, self.f3 = etdrk4_coefficients(self.k**2 - self.k**4, dt)
        centers = (np.linspace(0.0, length, n_actuators, endpoint=False)
                   if actuator_centers is None else np.asarray(actuator_centers, dtype=float))
        self.centers = centers
        Phi_ = np.column_stack([np.exp(-0.5 * (periodic_distance(self.x, c, length) / actuator_width)**2)
                                for c in centers])
        Phi_ -= Phi_.mean(axis=0, keepdims=True)
        Phi_ /= np.abs(Phi_).max(axis=0, keepdims=True)
        self.Phi, self.forcing_hat, self.n_action = Phi_, np.fft.fft(Phi_, axis=0), Phi_.shape[1]
        self.n_unstable_pairs = int(np.sum((self.k > 0) & (self.k < 1)))

    def _N(self, vh, a):
        v = np.fft.ifft(vh).real
        return (-0.5j * self.k * np.fft.fft(v**2)) * self.mask + self.forcing_hat @ np.asarray(a)

    def step(self, state, action, dtype=np.float32):
        v = np.asarray(state, dtype=np.float64); v = v - v.mean(); vh = np.fft.fft(v)
        for _ in range(self.n_sub):
            Nv = self._N(vh, action); sa = self.E2 * vh + self.Q * Nv; Na = self._N(sa, action)
            sb = self.E2 * vh + self.Q * Na; Nb = self._N(sb, action)
            sc = self.E2 * sa + self.Q * (2.0 * Nb - Nv); Nc = self._N(sc, action)
            vh = self.E * vh + self.f1 * Nv + 2.0 * self.f2 * (Na + Nb) + self.f3 * Nc
            vh[0] = 0.0
        out = np.fft.ifft(vh).real
        return (out - out.mean()).astype(dtype)

    def attractor(self, n_transient=100, n_keep=600, seed=1234):
        rng = np.random.default_rng(seed); u = np.zeros(self.n)
        for mode in range(1, 8):
            u += rng.normal(scale=0.5 / mode) * np.cos(2 * np.pi * mode * self.x / self.L)
            u += rng.normal(scale=0.5 / mode) * np.sin(2 * np.pi * mode * self.x / self.L)
        u = (u - u.mean()).astype(np.float32); z = np.zeros(self.n_action)
        for _ in range(n_transient):
            u = self.step(u, z)
        out = []
        for _ in range(n_keep):
            u = self.step(u, z); out.append(u.copy())
        return np.asarray(out, dtype=np.float32)


def summarise(df, label):
    """Row of the experiment tables from the per-initial-state DataFrame returned by ``evaluate``."""
    return {"configuration": label, "mean_cost": df.cost.mean(), "std_cost": df.cost.std(),
            "mean_effort_cost": df.effort_cost.mean(), "mean_final_energy": df.final_energy.mean(),
            "success_rate": df.success.mean(), "returned_to_chaos_rate": df.returned_to_chaos.mean()}


def reduced_linear_model(system, eps=1e-4):
    """Numerical Jacobians of the control-interval map around u=0, projected on the unstable Fourier pairs."""
    z_u, z_a = np.zeros(system.n), np.zeros(system.n_action)
    A = np.zeros((system.n, system.n)); B = np.zeros((system.n, system.n_action))
    for i in range(system.n):
        d = np.zeros(system.n); d[i] = eps
        A[:, i] = (system.step(d, z_a, np.float64) - system.step(-d, z_a, np.float64)) / (2 * eps)
    for j in range(system.n_action):
        d = np.zeros(system.n_action); d[j] = eps
        B[:, j] = (system.step(z_u, d, np.float64) - system.step(z_u, -d, np.float64)) / (2 * eps)
    basis = []
    for mode in range(1, system.n_unstable_pairs + 1):
        basis += [np.cos(2 * np.pi * mode * system.x / system.L), np.sin(2 * np.pi * mode * system.x / system.L)]
    basis, _ = np.linalg.qr(np.column_stack(basis))
    return basis.T @ A @ basis, basis.T @ B, basis


def design_lqr(system):
    """Local LQR on the unstable subspace: returns (controller state->action, closed-loop spectral radius)."""
    Ar, Br, basis = reduced_linear_model(system)
    Qr, Rr = np.eye(Ar.shape[0]) / system.n, 0.01 * np.eye(system.n_action)
    P = solve_discrete_are(Ar, Br, Qr, Rr)
    K = np.linalg.solve(Rr + Br.T @ P @ Br, Br.T @ P @ Ar)
    K_full = K @ basis.T
    radius = max(abs(np.linalg.eigvals(Ar - Br @ K)))
    return (lambda s: np.clip(-K_full @ np.asarray(s), -1.0, 1.0).astype(np.float32)), radius


def twin_divergence(system, u0, controller=None, steps=200, eps=1e-4, seed=0):
    """Distance between two trajectories that start eps apart (in the 2-norm)."""
    rng = np.random.default_rng(seed); d = rng.normal(size=system.n); d -= d.mean(); d *= eps / np.linalg.norm(d)
    ua = np.asarray(u0, np.float64); ub = ua + d; za = np.zeros(system.n_action)
    dist = [np.linalg.norm(ub - ua)]
    for _ in range(steps):
        aa = za if controller is None else controller(ua); ab = za if controller is None else controller(ub)
        ua = system.step(ua, aa, np.float64); ub = system.step(ub, ab, np.float64)
        dist.append(np.linalg.norm(ub - ua))
    return np.asarray(dist)


def lyapunov_slope(dist, dt=1.0, saturation=1.0):
    """Exponential growth rate fitted while the separation is still below `saturation`.

    ``dt`` is the control interval (1.0 in the notebook, where the default was the global ``control_dt``).
    """
    stop = int(np.argmax(dist > saturation)) if np.any(dist > saturation) else len(dist)
    if stop < 6:
        return np.nan
    t = np.arange(stop) * dt
    return float(np.polyfit(t, np.log(dist[:stop]), 1)[0])


def make_mpc(system, horizon=4, block=2, effort_weight=0.01, maxiter=25):
    """Short-horizon nonlinear MPC on the same ETDRK4 model, with move blocking and a warm start."""
    n_blocks, m = horizon // block, system.n_action
    plan = {"z": np.zeros(n_blocks * m)}

    def cost(z, u0):
        A = z.reshape(n_blocks, m); u = u0; c = 0.0
        for k in range(horizon):
            a = A[k // block]; u = system.step(u, a, np.float64)
            c += np.mean(u**2) + effort_weight * np.sum(a**2)
        return c

    def controller(state):
        warm = np.concatenate([plan["z"][m:], plan["z"][-m:]])
        res = minimize(cost, warm, args=(np.asarray(state, np.float64),), method="L-BFGS-B",
                       bounds=[(-1.0, 1.0)] * warm.size, options={"maxiter": maxiter})
        plan["z"] = res.x
        return res.x[:m].astype(np.float32)

    controller.reset = lambda: plan.update(z=np.zeros(n_blocks * m))
    return controller
