import numpy as np
from scipy.spatial import cKDTree

def _neighbor_lists(positions, radius):
    if positions.shape[0] == 0:
        return []
    tree = cKDTree(positions)
    return tree.query_ball_tree(tree, radius)

def separation(positions, radius):
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
    n, d = positions.shape
    accel = np.zeros((n, d))
    for i, nbrs in enumerate(_neighbor_lists(positions, radius)):
        others = [j for j in nbrs if j != i]
        if others:
            avg_vel = velocities[others].mean(axis=0)
            accel[i] = avg_vel - velocities[i]
    return accel

def cohesion(positions, radius):
    n, d = positions.shape
    accel = np.zeros((n, d))
    for i, nbrs in enumerate(_neighbor_lists(positions, radius)):
        others = [j for j in nbrs if j != i]
        if others:
            centroid = positions[others].mean(axis=0)
            accel[i] = centroid - positions[i]
    return accel

def obstacle_avoidance(positions, centers, radii, avoid_range):
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
    if not accels:
        raise ValueError("accels must not be empty")
    total = None
    for a, w in zip(accels, weights):
        contribution = a * w
        total = contribution if total is None else total + contribution
    return total

def clamp_speed(velocities, max_speed):
    out = velocities.copy()
    speeds = np.linalg.norm(out, axis=1)
    too_fast = speeds > max_speed
    out[too_fast] = out[too_fast] / speeds[too_fast, None] * max_speed
    return out

def neighbor_id_lists(positions, radius, include_self=False):
    out = []
    for i, nbrs in enumerate(_neighbor_lists(positions, radius)):
        ids = sorted(j for j in nbrs if include_self or j != i)
        out.append(ids)
    return out

def reflect_bounds(positions, velocities, size):
    pos = positions.copy()
    vel = velocities.copy()
    low = pos < 0.0
    pos[low] = -pos[low]
    vel[low] = np.abs(vel[low])
    high = pos > size
    pos[high] = 2.0 * size - pos[high]
    vel[high] = -np.abs(vel[high])
    return pos, vel

def resolve_obstacle_overlap(positions, velocities, centers, radii):
    # obstacle_avoidance is a steering force capped by its ramp, so a stronger
    # pull (food, or the flock centroid) can drag a boid straight through an
    # obstacle. This is the hard constraint that steering alone cannot give:
    # anything that ends a step inside an obstacle is placed back on the
    # surface and loses only the part of its velocity driving it deeper.
    pos = np.asarray(positions, dtype=float).copy()
    vel = np.asarray(velocities, dtype=float).copy()
    if centers is None:
        return pos, vel
    d = pos.shape[1]
    centers = np.asarray(centers, dtype=float).reshape(-1, d)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    for c, r in zip(centers, radii):
        offset = pos - c
        dist = np.linalg.norm(offset, axis=1)
        inside = dist < r
        if not inside.any():
            continue
        normal = np.zeros_like(offset)
        off_center = inside & (dist > 0.0)
        normal[off_center] = offset[off_center] / dist[off_center, None]
        # a boid exactly at the center has no outward direction of its own, so
        # pick a fixed one; an arbitrary choice here would break reproducibility
        dead_center = inside & (dist == 0.0)
        if dead_center.any():
            normal[dead_center, 0] = 1.0
        rows = np.where(inside)[0]
        pos[rows] = c + normal[rows] * r
        inward = np.einsum("ij,ij->i", vel[rows], normal[rows])
        vel[rows] -= np.minimum(inward, 0.0)[:, None] * normal[rows]
    return pos, vel

def enforce_min_speed(velocities, min_speed):
    out = velocities.copy()
    speeds = np.linalg.norm(out, axis=1)
    too_slow = (speeds < min_speed) & (speeds > 0)
    out[too_slow] = out[too_slow] / speeds[too_slow, None] * min_speed
    return out
