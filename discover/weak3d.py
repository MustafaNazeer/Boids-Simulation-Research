"""The weak form in 3D, for any feature function.

test_function and second_difference are copied from
meetings/september/28/identify/weak.py, which proves why they are exact.
discover/ does not import identify/, so the copy is deliberate.
"""

import numpy as np


def test_function(n):
    """A squared Hann window: it and its first difference vanish at both ends."""
    t = np.arange(n)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / (n - 1)))
    return hann ** 2


def second_difference(phi):
    """The discrete second difference, padded so it aligns with phi itself."""
    out = np.zeros_like(phi)
    out[1:-1] = phi[2:] - 2.0 * phi[1:-1] + phi[:-2]
    out[0] = phi[1] - 2.0 * phi[0]
    out[-1] = phi[-2] - 2.0 * phi[-1]
    return out


MAX_TERMS = 6


class BadFeatures(Exception):
    """A candidate's features returned something the fit cannot use."""


def frame_features(features, pos, vel, params, max_terms=MAX_TERMS):
    """Evaluate features on the agents present in one frame.

    Rows for absent agents come back as zeros; no equation is ever built for
    them because only focal agents, present in every frame, get equations.
    """
    present = ~np.isnan(pos[:, 0])
    if not present.any():
        raise BadFeatures("no agent present in a frame, so it has no center")
    p, v = pos[present], vel[present]
    g = np.asarray(features(p, v, p.mean(axis=0), params), dtype=float)
    if g.ndim != 3 or g.shape[0] != p.shape[0] or g.shape[1] != 3:
        raise BadFeatures("expected shape (%d, 3, K), got %s"
                          % (p.shape[0], g.shape))
    if g.shape[2] < 1:
        raise BadFeatures("features must return at least one term")
    if max_terms is not None and g.shape[2] > max_terms:
        raise BadFeatures("features may return at most %d terms, got %d"
                          % (max_terms, g.shape[2]))
    if not np.isfinite(g).all():
        raise BadFeatures("features returned values that are not finite")
    full = np.zeros((pos.shape[0], 3, g.shape[2]))
    full[present] = g
    return full


def window_equations(window, features, params, max_terms=MAX_TERMS):
    n = window.positions.shape[0]
    phi = test_function(n)
    d2 = second_difference(phi)
    frames = []
    for t in range(n):
        g = frame_features(features, window.positions[t],
                           window.velocities[t], params, max_terms)
        if frames and g.shape[2] != frames[0].shape[2]:
            raise BadFeatures("number of terms changed between frames")
        frames.append(g)
    G = np.stack(frames)[:, window.focal]
    target = window.positions[:, window.focal]
    y = np.einsum("t,tac->ac", d2, target) / (window.dt * window.dt)
    X = np.einsum("t,tack->ack", phi, G)
    return X.reshape(-1, X.shape[-1]), y.reshape(-1)


def equations(windows, features, params, max_terms=MAX_TERMS):
    """max_terms=None lifts the 6 term limit; only the fixed baseline library uses it."""
    Xs, ys = [], []
    for w in windows:
        X, y = window_equations(w, features, params, max_terms)
        if Xs and X.shape[1] != Xs[0].shape[1]:
            raise BadFeatures("number of terms changed between windows")
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)
