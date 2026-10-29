"""Tracking metrics: RMSE, MOTA/MOTP (CLEAR MOT), ID switches, OSPA, NEES."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


def ospa(X: np.ndarray, Y: np.ndarray, c: float = 50.0, p: int = 2) -> float:
    """OSPA distance between two point sets (Schuhmacher et al., 2008).

    Combines localization error and cardinality error (missed / false tracks)
    into one number in [0, c]. Lower is better.
    """
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c)
    if m > n:
        X, Y, m, n = Y, X, n, m
    D = np.minimum(np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2), c) ** p
    r, col = linear_sum_assignment(D)
    return float(((D[r, col].sum() + c**p * (n - m)) / n) ** (1.0 / p))


@dataclass
class MetricsAccumulator:
    match_dist: float = 30.0       # truth<->track match threshold [px]
    ospa_c: float = 50.0
    # running counts
    gt: int = 0
    fn: int = 0
    fp: int = 0
    idsw: int = 0
    dists: list = field(default_factory=list)
    nees: list = field(default_factory=list)
    # per-frame series for plots
    t: list = field(default_factory=list)
    ospa_series: list = field(default_factory=list)
    n_truth: list = field(default_factory=list)
    n_tracks: list = field(default_factory=list)
    rmse_series: list = field(default_factory=list)
    _last_match: dict = field(default_factory=dict)   # truth id -> track id
    assignments: list = field(default_factory=list)   # (t, truth id, track id)

    def update(self, t: float, truth: dict, tracks) -> None:
        tids = list(truth.keys())
        T = np.array([truth[i][:2] for i in tids]).reshape(-1, 2)
        E = np.array([trk.position for trk in tracks]).reshape(-1, 2)

        matched = []
        if len(T) and len(E):
            D = np.linalg.norm(T[:, None] - E[None], axis=2)
            r, c = linear_sum_assignment(np.where(D < self.match_dist, D, 1e6))
            matched = [(i, j) for i, j in zip(r, c) if D[i, j] < self.match_dist]

        self.gt += len(T)
        self.fn += len(T) - len(matched)
        self.fp += len(E) - len(matched)
        frame_err = []
        for i, j in matched:
            truth_id, trk = tids[i], tracks[j]
            if truth_id in self._last_match and self._last_match[truth_id] != trk.id:
                self.idsw += 1
            self._last_match[truth_id] = trk.id
            self.assignments.append((t, truth_id, trk.id))
            e = T[i] - trk.position
            frame_err.append(float(np.linalg.norm(e)))
            self.nees.append(float(e @ np.linalg.solve(trk.ekf.P[:2, :2], e)))
        self.dists += frame_err

        self.t.append(t)
        self.ospa_series.append(ospa(T, E, c=self.ospa_c))
        self.n_truth.append(len(T))
        self.n_tracks.append(len(E))
        self.rmse_series.append(float(np.sqrt(np.mean(np.square(frame_err)))) if frame_err else np.nan)

    def summary(self) -> dict:
        d = np.array(self.dists)
        nees = np.array(self.nees)
        lo, hi = chi2.ppf([0.025, 0.975], df=2)
        return {
            "frames": len(self.t),
            "gt_instances": self.gt,
            "MOTA": 1.0 - (self.fn + self.fp + self.idsw) / max(self.gt, 1),
            "MOTP_px": float(d.mean()) if len(d) else float("nan"),
            "pos_RMSE_px": float(np.sqrt(np.mean(d**2))) if len(d) else float("nan"),
            "recall": 1.0 - self.fn / max(self.gt, 1),
            "false_tracks": self.fp,
            "missed": self.fn,
            "id_switches": self.idsw,
            "mean_OSPA_px": float(np.mean(self.ospa_series)),
            "mean_NEES_pos": float(nees.mean()) if len(nees) else float("nan"),
            "NEES_in_95pct_band": float(np.mean((nees > lo) & (nees < hi))) if len(nees) else float("nan"),
        }
