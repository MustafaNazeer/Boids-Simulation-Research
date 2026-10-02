"""Windows of tracked agents, the unit every fit and score works on.

Real tracks break, so a window carries every agent seen in it (NaN where
unseen, so it can still act as a neighbor in the frames it exists) and a
separate list of focal agents tracked in every frame, which are the only
ones that get equations.

Whether an agent is tracked in a frame is judged from the x coordinate of
its position alone: NaN there means unseen. The data sources mark a missing
agent with NaN in every coordinate, so one coordinate is enough.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    source: str
    t0: int
    dt: float
    positions: np.ndarray
    velocities: np.ndarray
    focal: np.ndarray


def cut_windows(positions, velocities, dt, source, length, min_focal=3):
    out = []
    frames = positions.shape[0]
    for t0 in range(0, frames - length + 1, length):
        seg = slice(t0, t0 + length)
        seen = ~np.isnan(positions[seg, :, 0])
        keep = np.flatnonzero(seen.any(axis=0))
        focal = np.flatnonzero(seen[:, keep].all(axis=0))
        if len(focal) < min_focal:
            continue
        out.append(Window(source=source, t0=t0, dt=float(dt),
                          positions=positions[seg][:, keep].copy(),
                          velocities=velocities[seg][:, keep].copy(),
                          focal=focal))
    return out


def save_windows(windows, path):
    arrays = {"n": np.array(len(windows))}
    for i, w in enumerate(windows):
        arrays["source_%d" % i] = np.array(w.source)
        arrays["meta_%d" % i] = np.array([w.t0, w.dt])
        arrays["pos_%d" % i] = w.positions
        arrays["vel_%d" % i] = w.velocities
        arrays["focal_%d" % i] = w.focal
    np.savez(path, **arrays)


def load_windows(path):
    out = []
    with np.load(path) as z:
        for i in range(int(z["n"])):
            t0, dt = z["meta_%d" % i]
            out.append(Window(source=str(z["source_%d" % i]), t0=int(t0),
                              dt=float(dt), positions=z["pos_%d" % i],
                              velocities=z["vel_%d" % i],
                              focal=z["focal_%d" % i]))
    return out
