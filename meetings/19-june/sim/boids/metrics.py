import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def polarization(velocities):
    """Mean unit-velocity magnitude in [0, 1]. 1 means perfectly aligned."""
    speeds = np.linalg.norm(velocities, axis=1)
    moving = speeds > 0
    if not moving.any():
        return 0.0
    units = velocities[moving] / speeds[moving, None]
    return float(np.linalg.norm(units.mean(axis=0)))


def mean_nearest_neighbor_distance(positions):
    """Average distance from each agent to its single closest neighbor."""
    if len(positions) < 2:
        return 0.0
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)  # column 0 is self (distance 0)
    return float(dists[:, 1].mean())


def cluster_count(positions, threshold):
    """Number of connected components when agents within threshold are linked."""
    n = len(positions)
    if n == 0:
        return 0
    tree = cKDTree(positions)
    pairs = tree.query_pairs(threshold, output_type="ndarray")
    if len(pairs) == 0:
        return n
    data = np.ones(len(pairs))
    graph = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_components, _ = connected_components(graph, directed=False)
    return int(n_components)
