"""Run the tracker on a simulated star-field scene and save plots + metrics.

    python scripts/run_sim.py                          # defaults
    python scripts/run_sim.py --sensor range_bearing   # radar-style sensor
    python scripts/run_sim.py --clutter 20 --pd 0.8 --seed 3 --animate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mot import (CartesianMeasurement, CoordinatedTurnModel, MetricsAccumulator,  # noqa: E402
                 RangeBearingMeasurement, SimConfig, Tracker, TrackerConfig, simulate)
from mot import viz  # noqa: E402


def make_sensor(name: str):
    if name == "cartesian":
        return CartesianMeasurement(sigma=3.0)
    if name == "range_bearing":
        return RangeBearingMeasurement(origin=(500.0, 500.0), sigma_r=3.0, sigma_b=np.deg2rad(0.4))
    raise ValueError(name)


def run(sim_cfg: SimConfig, sensor_name: str = "cartesian", trk_cfg: TrackerConfig | None = None,
        record: bool = False):
    sensor = make_sensor(sensor_name)
    scen = simulate(sim_cfg, sensor)
    trk_cfg = trk_cfg or TrackerConfig(fov=(0.0, 0.0, sim_cfg.width, sim_cfg.height))
    tracker = Tracker(CoordinatedTurnModel(sigma_a=4.0, sigma_w=0.15), sensor, trk_cfg)
    acc = MetricsAccumulator()
    history = []
    for fr in scen.frames:
        tracker.step(fr.detections, sim_cfg.dt)
        reported = tracker.confirmed()
        acc.update(fr.t, fr.truth, reported)
        if record:
            history.append([(t.id, t.ekf.x.copy(), t.ekf.P.copy()) for t in reported])
    return scen, tracker, acc, history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", choices=["cartesian", "range_bearing"], default="cartesian")
    ap.add_argument("--targets", type=int, default=8)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--pd", type=float, default=0.9, help="probability of detection")
    ap.add_argument("--clutter", type=float, default=8.0, help="mean false alarms per frame")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--animate", action="store_true", help="also write tracking.gif")
    args = ap.parse_args()

    cfg = SimConfig(n_targets=args.targets, n_frames=args.frames, p_detect=args.pd,
                    clutter_rate=args.clutter, seed=args.seed)
    scen, tracker, acc, history = run(cfg, args.sensor, record=args.animate)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    viz.plot_trajectories(scen, tracker, out / "trajectories.png")
    viz.plot_metrics(acc, out / "metrics.png")
    viz.plot_consistency(acc, tracker.nis_log, out / "consistency.png")
    if args.animate:
        viz.animate(scen, history, tracker.sensor, out / "tracking.gif")

    summary = acc.summary()
    summary.update(sensor=args.sensor, pd=args.pd, clutter=args.clutter, seed=args.seed,
                   mean_NIS=float(np.mean(tracker.nis_log)),
                   tracks_confirmed=sum(t.id > 0 for t in tracker.deleted + tracker.tracks),
                   tracks_started=len(tracker.deleted) + len(tracker.tracks))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    width = max(map(len, summary))
    for k, v in summary.items():
        print(f"{k:<{width}}  {v:.3f}" if isinstance(v, float) else f"{k:<{width}}  {v}")
    print(f"\nplots written to {out}/")


if __name__ == "__main__":
    main()
