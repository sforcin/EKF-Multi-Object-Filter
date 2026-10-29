"""Monte Carlo sweep + ablations. Writes results/sweep.csv and results/sweep.md.

Variants
  CT-EKF + Hungarian : the full tracker
  CV-KF  + Hungarian : turn rate frozen at 0 -> a plain constant-velocity Kalman filter
                       (process noise tuned up to sigma_a = 20 so it can follow turns: a fair baseline)
  CT-EKF + greedy    : same filter, greedy nearest-neighbour association

    python scripts/sweep.py --seeds 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mot import (CartesianMeasurement, CoordinatedTurnModel, MetricsAccumulator,  # noqa: E402
                 SimConfig, Tracker, TrackerConfig, simulate)

VARIANTS = {
    "CT-EKF + Hungarian": dict(sigma_a=4.0, sigma_w=0.15, w0=0.3, assoc="hungarian"),
    "CV-KF + Hungarian": dict(sigma_a=20.0, sigma_w=0.0, w0=1e-9, assoc="hungarian"),
    "CT-EKF + greedy": dict(sigma_a=4.0, sigma_w=0.15, w0=0.3, assoc="greedy"),
}
SCENARIOS = [  # (p_detect, clutter per frame, number of targets)
    (0.95, 2.0, 10),
    (0.90, 10.0, 10),
    (0.90, 25.0, 10),
    (0.70, 10.0, 10),
    (0.90, 10.0, 30),   # dense: crossings stress data association
]
METRICS = ["MOTA", "pos_RMSE_px", "id_switches", "false_tracks", "mean_OSPA_px", "mean_NEES_pos"]


def run_one(pd, clutter, seed, v, n_targets=10):
    cfg = SimConfig(p_detect=pd, clutter_rate=clutter, seed=seed, n_targets=n_targets)
    sensor = CartesianMeasurement(sigma=3.0)
    scen = simulate(cfg, sensor)
    tcfg = TrackerConfig(fov=(0, 0, cfg.width, cfg.height), init_w_std=v["w0"], association=v["assoc"])
    trk = Tracker(CoordinatedTurnModel(sigma_a=v["sigma_a"], sigma_w=v["sigma_w"]), sensor, tcfg)
    acc = MetricsAccumulator()
    for fr in scen.frames:
        trk.step(fr.detections, cfg.dt)
        acc.update(fr.t, fr.truth, trk.confirmed())
    return acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    rows = []
    for pd, clutter, n in SCENARIOS:
        for name, v in VARIANTS.items():
            runs = [run_one(pd, clutter, 100 + s, v, n) for s in range(args.seeds)]
            row = {"targets": n, "Pd": pd, "clutter": clutter, "variant": name}
            for m in METRICS:
                row[m] = float(np.mean([r[m] for r in runs]))
            rows.append(row)
            print(f"n={n} Pd={pd:.2f} clutter={clutter:>4} {name:<20} "
                  + "  ".join(f"{m}={row[m]:.3f}" for m in METRICS))

    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hdr = ("| targets | Pd | clutter/frame | variant | MOTA | pos RMSE [px] | ID switches | "
           "false tracks | mean OSPA [px] | mean NEES (ideal 2) |")
    lines = [hdr, "|" + "---|" * 10]
    for r in rows:
        lines.append(f"| {r['targets']} | {r['Pd']:.2f} | {r['clutter']:.0f} | {r['variant']} | {r['MOTA']:.3f} | "
                     f"{r['pos_RMSE_px']:.2f} | {r['id_switches']:.1f} | {r['false_tracks']:.1f} | "
                     f"{r['mean_OSPA_px']:.2f} | {r['mean_NEES_pos']:.2f} |")
    (out / "sweep.md").write_text(
        f"Mean over {args.seeds} seeds, 300 frames at 10 Hz, 3 px measurement noise.\n\n"
        + "\n".join(lines) + "\n")
    print(f"\nwrote {out / 'sweep.md'}")


if __name__ == "__main__":
    main()
