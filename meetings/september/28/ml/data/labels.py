import numpy as np
from scipy.spatial import cKDTree

RELATIONS = ["separation", "cohesion", "alignment"]

_RADIUS_KEYS = {
    "separation": "separation_radius",
    "cohesion": "cohesion_radius",
    "alignment": "alignment_radius",
}


def radius_key(relation):
    return _RADIUS_KEYS[relation]


def edge_pairs(positions, radius):
    # query_pairs is inclusive at the radius and never returns a self pair,
    # which matches how rules._neighbor_lists builds the recorded neighbor
    # lists with query_ball_tree
    if positions.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.int64)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(float(radius), output_type="ndarray")
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.sort(pairs.astype(np.int64), axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]
