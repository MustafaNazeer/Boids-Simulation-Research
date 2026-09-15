import numpy as np


def roc_auc(scores, targets):
    """AUC via the Mann Whitney U statistic.

    Equal to the probability that a randomly chosen positive outranks a
    randomly chosen negative, with ties counted as half. Returns nan when one
    class is absent, because AUC is undefined there.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()

    n_pos = int((targets == 1).sum())
    n_neg = int((targets == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    # average the ranks inside each tied group so ties score exactly 0.5
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    rank_sum = ranks[targets == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def f1_at(scores, targets, threshold):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    predicted = scores >= threshold
    actual = targets == 1

    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    if tp == 0:
        return 0.0
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    return float(2 * precision * recall / (precision + recall))


def best_f1(scores, targets, n_thresholds=200):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size == 0:
        return 0.0, 0.0
    lo, hi = float(scores.min()), float(scores.max())
    if lo == hi:
        return f1_at(scores, targets, lo), lo
    best = (0.0, lo)
    for threshold in np.linspace(lo, hi, n_thresholds):
        score = f1_at(scores, targets, threshold)
        if score > best[0]:
            best = (score, float(threshold))
    return best
