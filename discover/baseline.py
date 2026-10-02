"""A WSINDy style baseline: a fixed library, sparse least squares, weak form.

fit searches the three radii with the same coordinate search the LLM
candidates get, so the comparison is fair. fit_fixed keeps the original fixed
percentile radii as a comparison point.

Sequentially thresholded least squares on column normalized equations, with
the threshold chosen by the same score the LLM candidates get, so the two are
judged identically. The radii come from the fit windows' own pairwise
distance percentiles, so nothing is tuned by hand to the answer.
"""

import numpy as np

from discover import verify, weak3d


def _frame_distances(windows):
    """The pairwise distance matrix of the agents present in each frame.

    Frames with fewer than two agents present have no pair, so they are
    skipped rather than yielding an empty or infinite distance.
    """
    found = False
    for w in windows:
        for t in range(w.positions.shape[0]):
            p = w.positions[t][~np.isnan(w.positions[t, :, 0])]
            if len(p) < 2:
                continue
            found = True
            yield np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    if not found:
        raise ValueError("no frame has two agents present, so no distance "
                         "scale can be measured")


def radii_from(windows):
    d = [dist[np.triu_indices(len(dist), 1)]
         for dist in _frame_distances(windows)]
    return [float(x) for x in np.percentile(np.concatenate(d), (25, 50, 75, 90))]


def radius_range(windows):
    nearest, pairs = [], []
    for dist in _frame_distances(windows):
        pairs.append(dist[np.triu_indices(len(dist), 1)])
        np.fill_diagonal(dist, np.inf)
        nearest.append(dist.min(axis=1))
    lo = 0.5 * float(np.percentile(np.concatenate(nearest), 5))
    hi = float(np.percentile(np.concatenate(pairs), 90))
    return lo, hi


def library():
    def features(pos, vel, center, params):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        k = np.exp(-(dist / params["r_rep"]) ** 2)
        rep = np.einsum("ij,ijk->ik", k, diff)
        near = (dist < params["r_rep"]) & (dist > 0)
        safe = np.where(near, dist, 1.0)
        sep_n = np.maximum(near.sum(axis=1), 1)[:, None]
        sep = np.einsum("ij,ijk->ik", near / safe ** 2, diff) / sep_n
        m = (dist < params["r_vel"]).astype(float)
        n = m.sum(axis=1)
        has = (n > 0)[:, None]
        nn = np.maximum(n, 1)[:, None]
        dv = np.where(has, (m @ vel) / nn - vel, 0.0)
        m = (dist < params["r_off"]).astype(float)
        n = m.sum(axis=1)
        has = (n > 0)[:, None]
        nn = np.maximum(n, 1)[:, None]
        off = np.where(has, (m @ pos) / nn - pos, 0.0)
        return np.stack([center - pos, -vel, rep, sep, dv, off], axis=-1)
    return features


def fixed_library(radii):
    def features(pos, vel, center, params):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        cols = [center - pos, -vel]
        for r in radii:
            k = np.exp(-(dist / r) ** 2)
            cols.append(np.einsum("ij,ijk->ik", k, diff))
            m = dist < r
            n = np.maximum(m.sum(axis=1), 1)[:, None]
            cols.append((m.astype(float) @ vel) / n - vel * (m.sum(axis=1) > 0)[:, None])
            cols.append((m.astype(float) @ pos) / n - pos * (m.sum(axis=1) > 0)[:, None])
        return np.stack(cols, axis=-1)
    return features


def _stlsq(X, y, threshold, iterations=10):
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    keep = np.arange(X.shape[1])
    for _ in range(iterations):
        w, _, _, _ = np.linalg.lstsq(Xs[:, keep], y, rcond=None)
        big = np.abs(w) >= threshold * np.abs(w).max()
        if big.all():
            break
        keep = keep[big]
    w, _, _, _ = np.linalg.lstsq(X[:, keep], y, rcond=None)
    return keep, w


def _best_threshold(Xf, yf, Xv, yv, thresholds, n_params):
    best = None
    for th in thresholds:
        keep, w = _stlsq(Xf, yf, th)
        r = yv - Xv[:, keep] @ w
        val_r2 = 1.0 - float(r @ r) / float(yv @ yv)
        s = verify.score(val_r2, len(keep), n_params)
        if best is None or s > best["score"]:
            best = {"kept": [int(k) for k in keep],
                    "weights": [float(x) for x in w], "val_r2": val_r2,
                    "score": s, "threshold": th}
    return best


def fit(fit_windows, val_windows, thresholds=(0.0, 0.01, 0.03, 0.1, 0.3)):
    lo, hi = radius_range(fit_windows)
    ranges = {"r_rep": (lo, hi), "r_vel": (lo, hi), "r_off": (lo, hi)}
    feats = library()
    found = verify.evaluate(fit_windows, val_windows, feats, ranges)
    Xf, yf = weak3d.equations(fit_windows, feats, found.params)
    Xv, yv = weak3d.equations(val_windows, feats, found.params)
    out = _best_threshold(Xf, yf, Xv, yv, thresholds, len(ranges))
    out["radii"] = found.params
    out["range"] = [lo, hi]
    return out


def fit_frozen(fit_windows, val_windows, test_windows,
               thresholds=(0.0, 0.01, 0.03, 0.1, 0.3)):
    """Choose everything on fit and val, then score the test windows once.

    fit picks the radii and the threshold by validation score, and its kept
    terms and weights come from the fit windows only. The test windows are
    only predicted with those frozen choices, so nothing is selected on them.
    """
    out = fit(fit_windows, val_windows, thresholds)
    Xt, yt = weak3d.equations(test_windows, library(), out["radii"])
    r = yt - Xt[:, out["kept"]] @ np.array(out["weights"])
    out["test_r2"] = 1.0 - float(r @ r) / float(yt @ yt)
    out["test_score"] = verify.score(out["test_r2"], len(out["kept"]),
                                     len(out["radii"]))
    return out


def fit_fixed(fit_windows, val_windows, thresholds=(0.0, 0.01, 0.03, 0.1, 0.3)):
    radii = radii_from(fit_windows)
    feats = fixed_library(radii)
    Xf, yf = weak3d.equations(fit_windows, feats, {}, max_terms=None)
    Xv, yv = weak3d.equations(val_windows, feats, {}, max_terms=None)
    out = _best_threshold(Xf, yf, Xv, yv, thresholds, 0)
    out["radii"] = radii
    return out
