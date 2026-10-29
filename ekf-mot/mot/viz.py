"""Plots for the README: trajectories, metrics over time, filter consistency."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

# Validated categorical palette (dark-surface steps) + neutral inks.
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#008300"]
SERIES_LIGHT = ["#2a78d6", "#eb6834"]
SPACE = "#0b0d14"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"


def _style_light(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_3)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_trajectories(scen, tracker, path, min_len: int = 10):
    """True paths vs. estimated tracks on a star field."""
    cfg = scen.cfg
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor(SPACE)
    ax.set_facecolor(SPACE)
    s = scen.stars
    ax.scatter(s[:, 0], s[:, 1], s=s[:, 2] * 2.5, c="white", alpha=0.35, linewidths=0)

    # clutter + detections from every 3rd frame (faint), converted to xy
    for fr in scen.frames[::3]:
        for z, cl in zip(fr.detections, fr.is_clutter):
            p = tracker.sensor.to_position(z)
            ax.plot(*p, ".", ms=1.6, color="#6b6a65" if cl else "#a9a8a0", alpha=0.6)

    # truth
    truth = {}
    for fr in scen.frames:
        for tid, x in fr.truth.items():
            truth.setdefault(tid, []).append(x[:2])
    for i, (tid, pts) in enumerate(truth.items()):
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], "--", color="white", lw=1.2, alpha=0.8,
                label="ground truth" if i == 0 else None)

    # estimates (confirmed / coasting portions only)
    all_tracks = tracker.deleted + tracker.tracks
    shown = [t for t in all_tracks
             if sum(h[3].value != "tentative" for h in t.history) >= min_len]
    for k, trk in enumerate(sorted(shown, key=lambda t: t.id)):
        pts = np.array([h[1][:2] for h in trk.history if h[3].value != "tentative"])
        col = SERIES_DARK[k % len(SERIES_DARK)]
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2, solid_capstyle="round",
                label="EKF track" if k == 0 else None)
        lx = np.clip(pts[-1, 0] + 8, 10, cfg.width - 50)
        ly = np.clip(pts[-1, 1] + 8, 10, cfg.height - 60)  # keep labels clear of the title
        ax.text(lx, ly, f"T{trk.id}", color="#f0efec", fontsize=8)

    ax.set_xlim(0, cfg.width)
    ax.set_ylim(0, cfg.height)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#f0efec")
    ax.set_title(f"EKF multi-object tracker: truth (dashed) vs. estimates  "
                 f"[Pd={cfg.p_detect}, clutter={cfg.clutter_rate}/frame]",
                 color="#f0efec", fontsize=10, loc="left", pad=12)
    ax.set_xlim(-10, cfg.width + 10)
    ax.set_ylim(-10, cfg.height + 10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SPACE)
    plt.close(fig)


def plot_metrics(acc, path):
    """OSPA over time and confirmed-track count vs. true target count."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), dpi=150, sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    t = np.array(acc.t)

    _style_light(a1)
    a1.plot(t, acc.ospa_series, color=SERIES_LIGHT[0], lw=1.5)
    a1.set_ylabel("OSPA [px]  (c = 50)", color=INK_2, fontsize=9)
    a1.set_title("OSPA distance per frame (lower is better)", color=INK, fontsize=10, loc="left")
    a1.set_ylim(0, acc.ospa_c * 1.05)

    _style_light(a2)
    a2.step(t, acc.n_truth, where="post", color=SERIES_LIGHT[0], lw=2, label="true targets")
    a2.step(t, acc.n_tracks, where="post", color=SERIES_LIGHT[1], lw=1.5, label="reported tracks")
    a2.set_ylabel("count", color=INK_2, fontsize=9)
    a2.set_xlabel("time [s]", color=INK_2, fontsize=9)
    a2.set_title("Cardinality: reported tracks vs. true targets", color=INK, fontsize=10, loc="left")
    a2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_consistency(acc, nis, path):
    """Position NEES and NIS histograms vs. the chi-square(2) density."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    xs = np.linspace(0, 12, 300)
    for ax, data, name in ((axes[0], acc.nees, "Position NEES (truth vs. estimate)"),
                           (axes[1], nis, "NIS (innovations at update)")):
        _style_light(ax)
        data = np.asarray(data)
        ax.hist(data[data < 12], bins=40, density=True, color=SERIES_LIGHT[0],
                alpha=0.85, edgecolor="#fcfcfb", linewidth=0.8, label="observed")
        ax.plot(xs, chi2.pdf(xs, 2), color=INK, lw=1.5, label=r"$\chi^2_2$ (ideal)")
        ax.set_title(f"{name}\nmean = {data.mean():.2f}  (ideal 2.0)", color=INK, fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def animate(scen, tracker_history, sensor, path, every: int = 2, fps: int = 15):
    """GIF of detections and confirmed tracks (for LinkedIn / the README)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    cfg = scen.cfg
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    fig.patch.set_facecolor(SPACE)
    frames = list(range(0, len(scen.frames), every))

    def draw(k):
        ax.clear()
        ax.set_facecolor(SPACE)
        s = scen.stars
        ax.scatter(s[:, 0], s[:, 1], s=s[:, 2] * 2, c="white", alpha=0.35, linewidths=0)
        fr = scen.frames[k]
        if len(fr.detections):
            P = np.array([sensor.to_position(z) for z in fr.detections])
            ax.scatter(P[:, 0], P[:, 1], s=10, c="#a9a8a0", marker="+", linewidths=0.8)
        trails = {}
        for past in tracker_history[max(0, k - 25):k + 1]:
            for tid, x, _ in past:
                trails.setdefault(tid, []).append(x[:2])
        for tid, x, Pcov in tracker_history[k]:
            col = SERIES_DARK[tid % len(SERIES_DARK)]
            tr = np.array(trails[tid])
            ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.5, alpha=0.8)
            ax.add_patch(_ellipse(x[:2], Pcov[:2, :2], col))
            ax.plot(*x[:2], "o", ms=4, color=col)
            ax.text(x[0] + 10, x[1] + 10, f"T{tid}", color="#f0efec", fontsize=7)
        ax.set_xlim(0, cfg.width); ax.set_ylim(0, cfg.height)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"t = {fr.t:5.1f} s   + detections   o tracks (3-sigma)",
                     color="#f0efec", fontsize=8, loc="left")

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _ellipse(mu, P, color, n_sigma=3.0):
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(P)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * n_sigma * np.sqrt(np.maximum(vals[::-1], 0))
    return Ellipse(mu, w, h, angle=angle, fill=False, color=color, lw=1)
