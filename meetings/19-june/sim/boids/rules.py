import numpy as np
from scipy.spatial import cKDTree


def _neighbor_lists(positions, radius):
    """Return, for each point, the list of indices within radius (includes self)."""
    if positions.shape[0] == 0:
        return []
    tree = cKDTree(positions)
    return tree.query_ball_tree(tree, radius)


def separation(positions, radius):
    """Steer away from neighbors within radius, weighted by inverse distance.

    Closer neighbors push harder (the steer contribution scales as 1/distance),
    which keeps the flock spaced out instead of collapsing to a point.
    Returns (N, D) accelerations.
    """
    n, d = positions.shape
    accel = np.zeros((n, d))
    for i, nbrs in enumerate(_neighbor_lists(positions, radius)):
        steer = np.zeros(d)
        count = 0
        for j in nbrs:
            if j == i:
                continue
            offset = positions[i] - positions[j]
            dist = np.linalg.norm(offset)
            if dist > 0:
                steer += offset / (dist * dist)
                count += 1
        if count:
            accel[i] = steer / count
    return accel


def alignment(positions, velocities, radius):
    """Steer toward the average velocity of neighbors. Returns (N, D)."""
    n, d = positions.shape
    accel = np.zeros((n, d))
    for i, nbrs in enumerate(_neighbor_lists(positions, radius)):
        others = [j for j in nbrs if j != i]
        if others:
            avg_vel = velocities[others].mean(axis=0)
            accel[i] = avg_vel - velocities[i]
    return accel


def cohesion(positions, radius):
    """Steer toward the centroid of neighbors. Returns (N, D)."""
    n, d = positions.shape
    accel = np.zeros((n, d))
    for i, nbrs in enumerate(_neighbor_lists(positions, radius)):
        others = [j for j in nbrs if j != i]
        if others:
            centroid = positions[others].mean(axis=0)
            accel[i] = centroid - positions[i]
    return accel


def obstacle_avoidance(positions, centers, radii, avoid_range):
    """Push away from circular obstacles, stronger the closer the boid is.

    centers: (M, D), radii: (M,), avoid_range: float buffer beyond each radius.
    """
    if avoid_range <= 0:
        raise ValueError("avoid_range must be positive")
    n, d = positions.shape
    accel = np.zeros((n, d))
    centers = np.asarray(centers, dtype=float).reshape(-1, d)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    for c, r in zip(centers, radii):
        offset = positions - c
        dist = np.linalg.norm(offset, axis=1)
        threshold = r + avoid_range
        mask = (dist < threshold) & (dist > 0)
        strength = (threshold - dist[mask]) / avoid_range
        accel[mask] += (offset[mask] / dist[mask, None]) * strength[:, None]
    return accel


def combine(accels, weights):
    """Weighted sum of a list of (N, D) acceleration arrays."""
    if not accels:
        raise ValueError("accels must be non-empty")
    total = None
    for a, w in zip(accels, weights):
        contribution = a * w
        total = contribution if total is None else total + contribution
    return total


def clamp_speed(velocities, max_speed):
    """Rescale any velocity whose magnitude exceeds max_speed."""
    out = velocities.copy()
    speeds = np.linalg.norm(out, axis=1)
    too_fast = speeds > max_speed
    out[too_fast] = out[too_fast] / speeds[too_fast, None] * max_speed
    return out


def neighbor_id_lists(positions, radius, include_self=False):
    """For each agent, the sorted list of neighbor indices within radius.

    Excludes the agent itself unless include_self is True. Reuses the same
    KD tree neighbor query as the flocking rules.
    """
    out = []
    for i, nbrs in enumerate(_neighbor_lists(positions, radius)):
        ids = sorted(j for j in nbrs if include_self or j != i)
        out.append(ids)
    return out
