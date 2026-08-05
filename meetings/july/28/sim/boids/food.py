import numpy as np
from scipy.spatial import cKDTree

NO_FOOD = -1

def seek(positions, food_positions, food_amounts, sensing_radius):

    positions = np.asarray(positions, dtype=float)
    n, d = positions.shape
    accel = np.zeros((n, d))
    food_ids = np.full(n, NO_FOOD, dtype=int)

    if food_positions is None:
        return accel, food_ids

    food_positions = np.asarray(food_positions, dtype=float)
    food_amounts = np.asarray(food_amounts, dtype=float)
    alive = np.where(food_amounts > 0)[0]
    if alive.size == 0:
        return accel, food_ids

    # query unbounded and filter with <= so the food graph agrees with the
    # three boids graphs at the boundary; distance_upper_bound is exclusive
    tree = cKDTree(food_positions[alive])
    dist, nearest = tree.query(positions)
    within = dist <= sensing_radius
    if not within.any():
        return accel, food_ids

    rows = np.where(within)[0]
    chosen = alive[nearest[within]]
    accel[rows] = food_positions[chosen] - positions[rows]
    food_ids[rows] = chosen

    return accel, food_ids
