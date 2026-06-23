"""Pre-planned agents: motion is a known function of the step index, not of
forces. Used standalone to verify the trajectory recorder, and dropped into the
flock as leaders the normal boids follow. Path math mirrors Angel Solis'
reference (github.com/asolis345/IsaacSimWBoids).
"""
import numpy as np

ORIENTATIONS = ("x", "y", "xy_pos", "xy_neg")

PLAN_CATALOG = [
    ("figure8", "x"), ("figure8", "y"), ("figure8", "xy_pos"), ("figure8", "xy_neg"),
    ("line", "x"), ("line", "y"), ("line", "xy_pos"), ("line", "xy_neg"),
]


def _orient(fx, fy, orientation, center):
    if orientation == "x":
        x, y = fx, fy
    elif orientation == "y":
        x, y = fy, fx
    elif orientation == "xy_pos":
        x = (fx - fy) / np.sqrt(2.0)
        y = (fx + fy) / np.sqrt(2.0)
    elif orientation == "xy_neg":
        x = (fx + fy) / np.sqrt(2.0)
        y = (-fx + fy) / np.sqrt(2.0)
    else:
        raise ValueError("unknown orientation: %r" % (orientation,))
    return np.column_stack([x, y]) + np.asarray(center, dtype=float)


def figure_eight(steps, orientation, amplitude, center):
    """Lissajous figure 8 path as a (steps, 2) array of positions."""
    t = np.linspace(0.0, 2.0 * np.pi, steps)
    fx = amplitude * np.sin(t)
    fy = amplitude * 0.5 * np.sin(t) * np.cos(t)
    return _orient(fx, fy, orientation, center)


def straight_line(steps, orientation, amplitude, center):
    """Back and forth straight line path as a (steps, 2) array of positions."""
    t = np.linspace(0.0, 2.0 * np.pi, steps)
    m = amplitude * np.sin(t)
    zero = np.zeros_like(m)
    if orientation == "x":
        return _orient(m, zero, "x", center)
    if orientation == "y":
        return _orient(m, zero, "y", center)
    if orientation == "xy_pos":
        x, y = m / np.sqrt(2.0), m / np.sqrt(2.0)
    elif orientation == "xy_neg":
        x, y = m / np.sqrt(2.0), -m / np.sqrt(2.0)
    else:
        raise ValueError("unknown orientation: %r" % (orientation,))
    return np.column_stack([x, y]) + np.asarray(center, dtype=float)


FACTORIES = {"figure8": figure_eight, "line": straight_line}


class PreplannedAgents:
    """Agents that follow precomputed paths at a constant reported speed.

    plans: list of (kind, orientation) with kind in {"figure8", "line"}.
    Position comes straight from the path array; velocity is the step to step
    path direction scaled to preplanned_speed (as in the reference).
    """

    def __init__(self, plans, steps, amplitude, center, preplanned_speed):
        paths = [FACTORIES[kind](steps, orient, amplitude, center)
                 for kind, orient in plans]
        self.paths = np.stack(paths)  # (M, steps, 2)
        self.steps = int(steps)
        self.preplanned_speed = float(preplanned_speed)

    @property
    def count(self):
        return self.paths.shape[0]

    def state_at(self, step):
        idx = step % self.steps
        nxt = (step + 1) % self.steps
        positions = self.paths[:, idx, :].copy()
        direction = self.paths[:, nxt, :] - positions
        dist = np.linalg.norm(direction, axis=1, keepdims=True)
        safe = np.maximum(dist, 1e-12)
        velocities = np.where(
            dist > 1e-9, direction / safe * self.preplanned_speed, 0.0)
        return positions, velocities
