import numpy as np
from scipy.spatial import cKDTree


def candidate_pairs(positions, k=None):
    """Pairs the model will score. `k=None` means every pair.

    Measured on the fixture on 2026-08-12, a tight k is actively harmful. At
    k=32 the cohesion recall ceiling is only 0.687 and 81 percent of the
    surviving candidates are positive, because being among a boid's 32
    nearest neighbors already almost implies a cohesion edge. That leaks
    "near means connected" harder than the radius cut this was meant to
    avoid, while discarding a third of the true edges. At k=128 recall is
    1.000 and the positive rate falls to 0.443, and the flock sizes the sweep
    produces (40 to 160) make that essentially all pairs anyway.

    So the default is all pairs. `k` is retained purely as a cost cap for
    flocks large enough that the quadratic pair count becomes a problem.
    """
    n = positions.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)

    if k is None or int(k) >= n - 1:
        i, j = np.triu_indices(n, k=1)
        return np.stack([i, j], axis=1).astype(np.int64)

    k_eff = min(int(k), n - 1)
    tree = cKDTree(positions)
    _, idx = tree.query(positions, k=k_eff + 1)
    idx = np.atleast_2d(idx)

    src = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    dst = idx.reshape(-1).astype(np.int64)

    keep = src != dst
    pairs = np.stack([src[keep], dst[keep]], axis=1)
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def to_directed(pairs):
    if pairs.shape[0] == 0:
        return np.zeros((2, 0), dtype=np.int64)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return np.stack([src, dst]).astype(np.int64)


def recall_ceiling(candidates_arr, truth):
    # the fraction of true edges the candidate set can possibly recover.
    # reported rather than assumed, because k caps recall from above
    if truth.shape[0] == 0:
        return 0.0
    cand = {tuple(p) for p in candidates_arr}
    hit = sum(1 for p in truth if tuple(p) in cand)
    return hit / float(truth.shape[0])
