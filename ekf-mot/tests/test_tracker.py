import numpy as np

from mot import (CartesianMeasurement, CoordinatedTurnModel, MetricsAccumulator, SimConfig,
                 Tracker, TrackerConfig, TrackState, ospa, simulate)
from mot.association import associate, gate_threshold
from mot.ekf import EKF


def _track_at(pos, std=3.0):
    class T:  # minimal stand-in with an EKF, enough for association
        pass
    t = T()
    t.ekf = EKF(np.r_[pos, 0, 0, 0], np.diag([std**2, std**2, 1, 1, 0.01]),
                CoordinatedTurnModel(), CartesianMeasurement())
    return t


def test_hungarian_beats_greedy():
    # Greedy nearest-neighbor would give det 0 to track 0 and leave track 1 unmatched.
    tracks = [_track_at([0.0, 0.0]), _track_at([10.0, 0.0])]
    dets = [np.array([6.0, 0.0]), np.array([-4.0, 0.0])]
    matches, ut, ud = associate(tracks, dets, gate_threshold(2, 0.999))
    assert sorted(matches) == [(0, 1), (1, 0)] and not ut and not ud


def test_gate_rejects_far_detection():
    matches, ut, ud = associate([_track_at([0.0, 0.0])], [np.array([200.0, 0.0])], gate_threshold())
    assert matches == [] and ut == [0] and ud == [0]


def test_track_lifecycle():
    trk = Tracker(CoordinatedTurnModel(), CartesianMeasurement())
    z = np.array([100.0, 100.0])
    for k in range(3):
        trk.step([z + [5.0 * k, 0.0]], 0.1)
    assert trk.tracks[0].state == TrackState.CONFIRMED and trk.tracks[0].id == 1
    trk.step([], 0.1)
    assert trk.tracks[0].state == TrackState.LOST
    for _ in range(trk.cfg.max_coast):
        trk.step([], 0.1)
    assert trk.tracks == []


def test_ospa_basics():
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert ospa(X, X) == 0.0
    assert ospa(X, np.empty((0, 2)), c=50) == 50
    assert np.isclose(ospa(X[:1], X, c=50, p=2), np.sqrt(50**2 / 2))


def test_end_to_end_clean_scene():
    """No clutter, perfect detection: tracker should be near-perfect."""
    cfg = SimConfig(n_targets=4, n_frames=150, p_detect=1.0, clutter_rate=0.0, seed=1)
    sensor = CartesianMeasurement(sigma=3.0)
    scen = simulate(cfg, sensor)
    tracker = Tracker(CoordinatedTurnModel(sigma_a=4.0, sigma_w=0.15), sensor,
                      TrackerConfig(fov=(0, 0, cfg.width, cfg.height)))
    acc = MetricsAccumulator()
    for fr in scen.frames:
        tracker.step(fr.detections, cfg.dt)
        acc.update(fr.t, fr.truth, tracker.confirmed())
    s = acc.summary()
    # Not 1.0: each track needs 3 frames to confirm, and coasts a few frames after
    # its target disappears mid-scene (the tracker can't tell death from a miss).
    assert s["recall"] > 0.95
    assert s["MOTA"] > 0.90
    assert s["pos_RMSE_px"] < 3.0           # better than the 3 px raw measurement noise
    assert s["id_switches"] == 0
