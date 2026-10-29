"""Motion and measurement models for the EKF.

State vector (5,):  x = [px, py, vx, vy, w]
    px, py : position            [px]
    vx, vy : velocity            [px / s]
    w      : turn rate           [rad / s]   (w = 0 -> straight line)

This is the "coordinated turn" (CT) model from Bar-Shalom, Li & Kirubarajan,
"Estimation with Applications to Tracking and Navigation", ch. 11. The
dynamics are nonlinear in w, which is why we need an *Extended* Kalman Filter.
"""
from __future__ import annotations

import numpy as np

SMALL_W = 1e-3  # below this turn rate we use a Taylor series (avoids 0/0 and cancellation)


# ----------------------------------------------------------------------------
# Motion model
# ----------------------------------------------------------------------------
class CoordinatedTurnModel:
    """Nearly-coordinated-turn motion model with white acceleration noise."""

    dim_x = 5

    def __init__(self, sigma_a: float = 2.0, sigma_w: float = 0.05):
        self.sigma_a = sigma_a  # std of random linear acceleration [px/s^2]
        self.sigma_w = sigma_w  # std of random turn-rate change    [rad/s^2]

    @staticmethod
    def _coeffs(w: float, dt: float):
        """a = sin(wT)/w, b = (1 - cos(wT))/w and their derivatives in w."""
        if abs(w) < SMALL_W:
            # Taylor expansion around w = 0 keeps everything finite and accurate.
            a = dt - w**2 * dt**3 / 6.0
            b = 0.5 * w * dt**2 - w**3 * dt**4 / 24.0
            da = -w * dt**3 / 3.0
            db = 0.5 * dt**2 - w**2 * dt**4 / 8.0
        else:
            s, c = np.sin(w * dt), np.cos(w * dt)
            a = s / w
            b = 2.0 * np.sin(0.5 * w * dt) ** 2 / w   # = (1 - cos wT)/w, without cancellation
            da = (dt * c * w - s) / w**2
            db = (dt * s * w - (1.0 - c)) / w**2
        return a, b, da, db

    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Propagate the state dt seconds forward (noise-free)."""
        px, py, vx, vy, w = x
        a, b, _, _ = self._coeffs(w, dt)
        s, c = np.sin(w * dt), np.cos(w * dt)
        return np.array([
            px + a * vx - b * vy,
            py + b * vx + a * vy,
            c * vx - s * vy,
            s * vx + c * vy,
            w,
        ])

    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Jacobian df/dx evaluated at x (5x5)."""
        _, _, vx, vy, w = x
        a, b, da, db = self._coeffs(w, dt)
        s, c = np.sin(w * dt), np.cos(w * dt)
        J = np.eye(5)
        J[0, 2], J[0, 3], J[0, 4] = a, -b, da * vx - db * vy
        J[1, 2], J[1, 3], J[1, 4] = b, a, db * vx + da * vy
        J[2, 2], J[2, 3], J[2, 4] = c, -s, -dt * s * vx - dt * c * vy
        J[3, 2], J[3, 3], J[3, 4] = s, c, dt * c * vx - dt * s * vy
        return J

    def Q(self, dt: float) -> np.ndarray:
        """Discrete process noise (white-noise acceleration + turn-rate noise)."""
        q = self.sigma_a**2
        Q = np.zeros((5, 5))
        for p, v in ((0, 2), (1, 3)):
            Q[p, p] = dt**4 / 4 * q
            Q[p, v] = Q[v, p] = dt**3 / 2 * q
            Q[v, v] = dt**2 * q
        Q[4, 4] = (self.sigma_w * dt) ** 2
        return Q


# ----------------------------------------------------------------------------
# Measurement models
# ----------------------------------------------------------------------------
class CartesianMeasurement:
    """Camera-style detection: z = [px, py] + noise (linear in the state)."""

    dim_z = 2

    def __init__(self, sigma: float = 3.0):
        self.sigma = sigma
        self.R = np.eye(2) * sigma**2

    def h(self, x: np.ndarray) -> np.ndarray:
        return x[:2].copy()

    def H(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((2, 5))
        H[0, 0] = H[1, 1] = 1.0
        return H

    def residual(self, z: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
        return z - z_pred

    def simulate(self, pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return pos + rng.normal(0.0, self.sigma, size=2)

    def to_position(self, z: np.ndarray) -> np.ndarray:
        return z.copy()

    def position_cov(self, z: np.ndarray) -> np.ndarray:
        return self.R.copy()


class RangeBearingMeasurement:
    """Radar-style detection from a sensor at `origin`: z = [range, bearing].

    Nonlinear in the state, so the EKF linearizes h(x) around the prediction.
    """

    dim_z = 2

    def __init__(self, origin=(500.0, 500.0), sigma_r: float = 3.0,
                 sigma_b: float = np.deg2rad(0.5)):
        self.origin = np.asarray(origin, dtype=float)
        self.sigma_r, self.sigma_b = sigma_r, sigma_b
        self.R = np.diag([sigma_r**2, sigma_b**2])

    def h(self, x: np.ndarray) -> np.ndarray:
        dx, dy = x[0] - self.origin[0], x[1] - self.origin[1]
        return np.array([np.hypot(dx, dy), np.arctan2(dy, dx)])

    def H(self, x: np.ndarray) -> np.ndarray:
        dx, dy = x[0] - self.origin[0], x[1] - self.origin[1]
        r2 = max(dx * dx + dy * dy, 1e-9)
        r = np.sqrt(r2)
        H = np.zeros((2, 5))
        H[0, 0], H[0, 1] = dx / r, dy / r
        H[1, 0], H[1, 1] = -dy / r2, dx / r2
        return H

    def residual(self, z: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
        y = z - z_pred
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi  # wrap bearing to [-pi, pi)
        return y

    def simulate(self, pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z = self.h(np.r_[pos, 0, 0, 0])
        z += rng.normal(0.0, [self.sigma_r, self.sigma_b])
        return z

    def to_position(self, z: np.ndarray) -> np.ndarray:
        r, b = z
        return self.origin + r * np.array([np.cos(b), np.sin(b)])

    def position_cov(self, z: np.ndarray) -> np.ndarray:
        """Covariance of the converted position (first-order)."""
        r, b = z
        J = np.array([[np.cos(b), -r * np.sin(b)],
                      [np.sin(b), r * np.cos(b)]])
        return J @ self.R @ J.T
