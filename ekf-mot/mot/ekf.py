"""Extended Kalman Filter for one target."""
from __future__ import annotations

import numpy as np


class EKF:
    """EKF with a pluggable motion model and measurement model.

    predict:  x <- f(x),            P <- F P F^T + Q
    update:   y = z - h(x),         S = H P H^T + R
              K = P H^T S^-1,       x <- x + K y
              P <- (I - K H) P (I - K H)^T + K R K^T     (Joseph form)
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray, motion, sensor):
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.motion = motion
        self.sensor = sensor

    # -- prediction -----------------------------------------------------------
    def predict(self, dt: float) -> None:
        F = self.motion.F(self.x, dt)  # linearize at the *prior* mean
        self.x = self.motion.f(self.x, dt)
        self.P = F @ self.P @ F.T + self.motion.Q(dt)
        self.P = 0.5 * (self.P + self.P.T)  # keep it symmetric

    # -- measurement ----------------------------------------------------------
    def innovation(self, z: np.ndarray):
        """Return (y, S, H) for a candidate measurement z."""
        H = self.sensor.H(self.x)
        y = self.sensor.residual(z, self.sensor.h(self.x))
        S = H @ self.P @ H.T + self.sensor.R
        return y, S, H

    def mahalanobis2(self, z: np.ndarray) -> float:
        """Squared Mahalanobis distance d^2 = y^T S^-1 y (used for gating)."""
        y, S, _ = self.innovation(z)
        return float(y @ np.linalg.solve(S, y))

    def update(self, z: np.ndarray) -> float:
        """Fuse measurement z. Returns its normalized innovation squared (NIS)."""
        y, S, H = self.innovation(z)
        K = np.linalg.solve(S, H @ self.P).T  # = P H^T S^-1 (S is symmetric)
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.sensor.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return float(y @ np.linalg.solve(S, y))
