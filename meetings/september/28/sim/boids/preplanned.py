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
    t = np.linspace(0.0, 2.0 * np.pi, steps)
    fx = amplitude * np.sin(t)
    fy = amplitude * 0.5 * np.sin(t) * np.cos(t)
    return _orient(fx, fy, orientation, center)

def straight_line(steps, orientation, amplitude, center):
    t = np.linspace(0.0, 2.0 * np.pi, steps)
    m = amplitude * np.sin(t)
    zero = np.zeros_like(m)
    return _orient(m, zero, orientation, center)

FACTORIES = {"figure8": figure_eight, "line": straight_line}

class PreplannedAgents:

    def __init__(self, plans, steps, amplitude, center, preplanned_speed):
        paths = [FACTORIES[kind](steps, orient, amplitude, center)
                 for kind, orient in plans]
        self.paths = np.stack(paths)
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
