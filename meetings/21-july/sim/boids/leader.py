import numpy as np
from scipy.spatial import cKDTree

NO_LEADER = -1

def follow(positions, leader_mask, follow_radius):

    positions = np.asarray(positions, dtype=float)
    n, d = positions.shape
    accel = np.zeros((n, d))

    leader_ids = np.full(n, NO_LEADER, dtype=int)
    mask = np.asarray(leader_mask, dtype=bool)
    leaders = np.where(mask)[0]
    followers = np.where(~mask)[0]

    if leaders.size == 0 or followers.size == 0:
        return accel, leader_ids

    tree = cKDTree(positions[leaders])
    dist, nearest = tree.query(positions[followers])
    within = dist <= follow_radius

    if not within.any():
        return accel, leader_ids

    rows = followers[within]
    chosen = leaders[nearest[within]]
    accel[rows] = positions[chosen] - positions[rows]
    leader_ids[rows] = chosen

    return accel, leader_ids

def wander(headings, leader_mask, rng, wander_strength):

    headings = np.asarray(headings, dtype=float).copy()
    mask = np.asarray(leader_mask, dtype=bool)
    accel = np.zeros((mask.shape[0], 2))

    leaders = np.where(mask)[0]

    if leaders.size == 0:
        return accel, headings

    headings[leaders] += rng.uniform(-wander_strength, wander_strength,
                                     size=leaders.size)

    accel[leaders, 0] = np.cos(headings[leaders])
    accel[leaders, 1] = np.sin(headings[leaders])

    return accel, headings

def initial_headings(n, rng):
    return rng.uniform(0.0, 2.0 * np.pi, size=n)