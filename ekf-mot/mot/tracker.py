"""Multi-object tracker: EKF per track + Hungarian association + track lifecycle.

Track lifecycle
---------------
    new detection --> TENTATIVE --(confirm_hits hits)--> CONFIRMED
    TENTATIVE  --(tentative_max_misses misses in a row)--> deleted
    CONFIRMED  --(miss)--> LOST   (coasts on prediction only)
    LOST       --(hit)---> CONFIRMED
    LOST       --(max_coast misses in a row)--> deleted

The `Tracker.step(detections, dt)` interface is sensor-agnostic: Level 2 feeds
it OpenCV blob centroids instead of simulated detections.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .association import associate, gate_threshold
from .ekf import EKF


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass
class TrackerConfig:
    gate_prob: float = 0.99          # chi-square gate probability
    association: str = "hungarian"   # or "greedy" (baseline for the ablation)
    confirm_hits: int = 3            # hits needed to confirm a tentative track
    tentative_max_misses: int = 2    # consecutive misses that kill a tentative track
    max_coast: int = 6               # consecutive misses before a LOST track is deleted
    init_vel_std: float = 40.0       # prior std on velocity for a new track [px/s]
    init_w_std: float = 0.3          # prior std on turn rate for a new track [rad/s]
    max_pos_std: float = 80.0        # delete tracks whose position std blows up [px]
    fov: tuple | None = None         # (xmin, ymin, xmax, ymax): drop tracks that leave the sensor's view


@dataclass
class Track:
    id: int                          # -1 while tentative; assigned on confirmation
    ekf: EKF
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    misses: int = 0                  # consecutive misses
    age: int = 1
    history: list = field(default_factory=list)  # (t, x, P, state)

    @property
    def position(self) -> np.ndarray:
        return self.ekf.x[:2]


class Tracker:
    def __init__(self, motion, sensor, config: TrackerConfig | None = None):
        self.motion = motion
        self.sensor = sensor
        self.cfg = config or TrackerConfig()
        self.gate = gate_threshold(sensor.dim_z, self.cfg.gate_prob)
        self.tracks: list[Track] = []
        self.deleted: list[Track] = []
        self._ids = itertools.count(1)
        self.t = 0.0
        self.nis_log: list[float] = []  # NIS of every confirmed-track update

    # ------------------------------------------------------------------------
    def _new_track(self, z: np.ndarray) -> Track:
        pos = self.sensor.to_position(z)
        P0 = np.zeros((5, 5))
        P0[:2, :2] = self.sensor.position_cov(z)
        P0[2, 2] = P0[3, 3] = self.cfg.init_vel_std**2
        P0[4, 4] = self.cfg.init_w_std**2
        x0 = np.array([pos[0], pos[1], 0.0, 0.0, 0.0])
        return Track(id=-1, ekf=EKF(x0, P0, self.motion, self.sensor))

    def _confirm(self, trk: Track) -> None:
        trk.state = TrackState.CONFIRMED
        if trk.id < 0:  # IDs only for confirmed tracks, so they stay small and readable
            trk.id = next(self._ids)

    def _delete(self, trk: Track) -> None:
        self.tracks.remove(trk)
        self.deleted.append(trk)

    # ------------------------------------------------------------------------
    def step(self, detections, dt: float) -> list[Track]:
        """Process one frame of detections (list/array of z vectors)."""
        detections = [np.asarray(z, dtype=float) for z in detections]
        self.t += dt

        # 1. Predict every track forward.
        for trk in self.tracks:
            trk.ekf.predict(dt)
            trk.age += 1

        # 2. Gate + associate (Hungarian on Mahalanobis cost).
        matches, unmatched_t, unmatched_d = associate(self.tracks, detections, self.gate,
                                                     self.cfg.association)

        # 3. Update matched tracks.
        for ti, di in matches:
            trk = self.tracks[ti]
            nis = trk.ekf.update(detections[di])
            trk.hits += 1
            trk.misses = 0
            if trk.state == TrackState.TENTATIVE and trk.hits >= self.cfg.confirm_hits:
                self._confirm(trk)
            elif trk.state == TrackState.LOST:
                self._confirm(trk)
            if trk.state == TrackState.CONFIRMED:
                self.nis_log.append(nis)

        # 4. Handle misses (lifecycle) -- iterate over a copy since we delete.
        missed = [self.tracks[i] for i in unmatched_t]
        for trk in missed:
            trk.misses += 1
            if trk.state == TrackState.TENTATIVE:
                if trk.misses >= self.cfg.tentative_max_misses:
                    self._delete(trk)
            else:
                trk.state = TrackState.LOST
                if trk.misses >= self.cfg.max_coast:
                    self._delete(trk)

        # Delete tracks whose uncertainty has exploded (filter has lost it)
        # or that have coasted out of the sensor's field of view.
        for trk in list(self.tracks):
            if np.sqrt(np.max(np.linalg.eigvalsh(trk.ekf.P[:2, :2]))) > self.cfg.max_pos_std:
                self._delete(trk)
            elif self.cfg.fov is not None:
                x0, y0, x1, y1 = self.cfg.fov
                px, py = trk.position
                if not (x0 <= px <= x1 and y0 <= py <= y1):
                    self._delete(trk)

        # 5. Unmatched detections start new tentative tracks.
        for di in unmatched_d:
            self.tracks.append(self._new_track(detections[di]))

        for trk in self.tracks:
            trk.history.append((self.t, trk.ekf.x.copy(), trk.ekf.P.copy(), trk.state))
        return self.tracks

    def confirmed(self) -> list[Track]:
        """Tracks we would report downstream (confirmed or briefly coasting)."""
        return [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
