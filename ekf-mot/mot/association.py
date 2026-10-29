"""Global nearest-neighbor data association with gating + Hungarian algorithm."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

INFEASIBLE = 1e6


def gate_threshold(dim_z: int = 2, prob: float = 0.99) -> float:
    """Chi-square gate: a true detection falls inside with probability `prob`."""
    return float(chi2.ppf(prob, df=dim_z))


def build_cost_matrix(tracks, detections, gate: float) -> np.ndarray:
    """cost[i, j] = d^2 + ln|S| for track i vs detection j, or INFEASIBLE.

    Adding ln|S| penalizes uncertain tracks, so a sharp confirmed track wins a
    detection over a vague new one at the same Mahalanobis distance.
    """
    cost = np.full((len(tracks), len(detections)), INFEASIBLE)
    for i, trk in enumerate(tracks):
        for j, z in enumerate(detections):
            y, S, _ = trk.ekf.innovation(z)
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 <= gate:
                cost[i, j] = d2 + np.log(np.linalg.det(S))
    return cost


def greedy_assignment(cost: np.ndarray):
    """Baseline: repeatedly take the cheapest remaining pair (local, not optimal)."""
    rows, cols = [], []
    c = cost.copy()
    while c.size and c.min() < INFEASIBLE:
        r, k = np.unravel_index(np.argmin(c), c.shape)
        rows.append(r)
        cols.append(k)
        c[r, :] = INFEASIBLE
        c[:, k] = INFEASIBLE
    return np.array(rows, int), np.array(cols, int)


def associate(tracks, detections, gate: float, method: str = "hungarian"):
    """Return (matches, unmatched_track_idx, unmatched_det_idx)."""
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = build_cost_matrix(tracks, detections, gate)
    if method == "hungarian":
        rows, cols = linear_sum_assignment(cost)  # globally optimal one-to-one
    elif method == "greedy":
        rows, cols = greedy_assignment(cost)
    else:
        raise ValueError(method)

    matches = []
    used_t, used_d = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] < INFEASIBLE:  # Hungarian may pair outside the gate; drop those
            matches.append((r, c))
            used_t.add(r)
            used_d.add(c)
    unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_d = [j for j in range(len(detections)) if j not in used_d]
    return matches, unmatched_t, unmatched_d
