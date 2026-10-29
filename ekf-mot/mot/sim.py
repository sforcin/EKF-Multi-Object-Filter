"""2D star-field scenario: maneuvering targets, missed detections, clutter."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import CoordinatedTurnModel


@dataclass
class SimConfig:
    width: float = 1000.0
    height: float = 1000.0
    dt: float = 0.1                 # 10 Hz frame rate
    n_frames: int = 300             # 30 s
    n_targets: int = 8
    speed_range: tuple = (25.0, 70.0)        # px/s
    max_turn_rate: float = 0.35              # rad/s
    maneuver_prob: float = 0.01              # per-frame chance a target changes turn rate
    p_detect: float = 0.9                    # probability of detection
    clutter_rate: float = 8.0                # mean false alarms per frame (Poisson)
    n_stars: int = 400                       # background stars (visual only)
    seed: int = 7


@dataclass
class Frame:
    t: float
    truth: dict                     # target id -> true state (5,)
    detections: np.ndarray          # (k, dim_z)
    is_clutter: np.ndarray          # (k,) bool, for plotting only


@dataclass
class Scenario:
    cfg: SimConfig
    frames: list = field(default_factory=list)
    stars: np.ndarray | None = None


def _spawn(rng, cfg: SimConfig) -> np.ndarray:
    """New target on the inner 80% of the scene heading roughly inward."""
    m = 0.1
    px = rng.uniform(m * cfg.width, (1 - m) * cfg.width)
    py = rng.uniform(m * cfg.height, (1 - m) * cfg.height)
    center = np.array([cfg.width / 2, cfg.height / 2])
    heading = np.arctan2(*(center - [px, py])[::-1]) + rng.normal(0, 0.8)
    speed = rng.uniform(*cfg.speed_range)
    w = rng.choice([0.0, rng.uniform(-cfg.max_turn_rate, cfg.max_turn_rate)])
    return np.array([px, py, speed * np.cos(heading), speed * np.sin(heading), w])


def simulate(cfg: SimConfig, sensor, rng: np.random.Generator | None = None) -> Scenario:
    rng = rng or np.random.default_rng(cfg.seed)
    truth_model = CoordinatedTurnModel(sigma_a=1.0, sigma_w=0.0)

    # Staggered births so the tracker has to initiate tracks mid-run.
    births = np.sort(rng.integers(0, cfg.n_frames // 2, size=cfg.n_targets))
    births[: cfg.n_targets // 2] = 0
    lifetimes = rng.integers(cfg.n_frames // 3, cfg.n_frames, size=cfg.n_targets)

    alive: dict[int, np.ndarray] = {}
    scen = Scenario(cfg=cfg)
    scen.stars = np.c_[rng.uniform(0, cfg.width, cfg.n_stars),
                       rng.uniform(0, cfg.height, cfg.n_stars),
                       rng.uniform(0.2, 1.0, cfg.n_stars)]  # x, y, brightness

    for k in range(cfg.n_frames):
        t = (k + 1) * cfg.dt
        # births / deaths
        for tid in range(cfg.n_targets):
            if births[tid] == k:
                alive[tid] = _spawn(rng, cfg)
            if tid in alive and k >= births[tid] + lifetimes[tid]:
                del alive[tid]

        # propagate truth (with occasional maneuvers and a little accel noise)
        for tid, x in list(alive.items()):
            if rng.random() < cfg.maneuver_prob:
                x[4] = rng.choice([0.0, rng.uniform(-cfg.max_turn_rate, cfg.max_turn_rate)])
            x = truth_model.f(x, cfg.dt)
            x[2:4] += rng.normal(0, truth_model.sigma_a * cfg.dt, size=2)
            if not (0 <= x[0] <= cfg.width and 0 <= x[1] <= cfg.height):
                del alive[tid]  # left the field of view
                continue
            alive[tid] = x

        # detections: each target seen with prob p_detect, plus Poisson clutter
        dets, clutter = [], []
        for x in alive.values():
            if rng.random() < cfg.p_detect:
                dets.append(sensor.simulate(x[:2], rng))
                clutter.append(False)
        for _ in range(rng.poisson(cfg.clutter_rate)):
            p = np.array([rng.uniform(0, cfg.width), rng.uniform(0, cfg.height)])
            dets.append(sensor.h(np.r_[p, 0, 0, 0]))
            clutter.append(True)

        order = rng.permutation(len(dets))  # don't leak truth order to the tracker
        dets = np.array(dets).reshape(-1, sensor.dim_z)[order]
        scen.frames.append(Frame(t=t, truth={i: v.copy() for i, v in alive.items()},
                                 detections=dets, is_clutter=np.array(clutter, bool)[order]))
    return scen
