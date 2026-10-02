"""Fit a candidate rule's weights and parameters, and score it.

Weights come from least squares on the weak form equations. Tunable
parameters come from a coarse to fine coordinate search chosen by fit set
R^2, the same deterministic style identify/weak.py estimate uses. The score
uses validation R^2, so a parameter can never be tuned on the windows that
judge it.
"""

from dataclasses import dataclass

import numpy as np

from discover import weak3d

PENALTY_TERM = 0.002
PENALTY_PARAM = 0.002
MAX_PARAMS = 3


class BadCandidate(Exception):
    """The candidate declared something the search cannot use."""


@dataclass
class Fit:
    weights: list
    params: dict
    fit_r2: float
    val_r2: float
    n_terms: int
    score: float
    ranges: dict


def score(val_r2, n_terms, n_params):
    return val_r2 - PENALTY_TERM * n_terms - PENALTY_PARAM * n_params


def _r2(X, y, w):
    r = y - X @ w
    return 1.0 - float(r @ r) / float(y @ y)


def _fit_r2(fit_windows, features, params):
    X, y = weak3d.equations(fit_windows, features, params)
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return w, _r2(X, y, w)


def _check_ranges(param_ranges):
    if len(param_ranges) > MAX_PARAMS:
        raise BadCandidate("at most %d tunable parameters, got %d"
                           % (MAX_PARAMS, len(param_ranges)))
    for name, bounds in param_ranges.items():
        try:
            lo, hi = (float(b) for b in bounds)
        except (TypeError, ValueError):
            raise BadCandidate("parameter %s needs a (low, high) pair of "
                               "numbers, got %r" % (name, bounds))
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise BadCandidate("parameter %s has an invalid range %r"
                               % (name, bounds))


def is_distance(name):
    """Parameters named r or r_<name> are distances."""
    return name == "r" or name.startswith("r_")


def _apply_floor(ranges, radius_floor):
    if radius_floor is None:
        return ranges
    floored = {}
    for name, (lo, hi) in ranges.items():
        if is_distance(name) and lo < radius_floor:
            if radius_floor >= hi:
                raise BadCandidate(
                    "distance parameter %s has range (%g, %g), which lies "
                    "wholly at or below the minimum distance %g for this data"
                    % (name, lo, hi, radius_floor))
            lo = float(radius_floor)
        floored[name] = (lo, hi)
    return floored


def evaluate(fit_windows, val_windows, features, param_ranges,
             levels=4, points=5, radius_floor=None):
    """Fit and score one candidate.

    With radius_floor set, a distance parameter (see is_distance) whose
    declared lower bound is below the floor is searched from the floor
    instead. Fit.ranges records the ranges actually searched.
    """
    _check_ranges(param_ranges)
    ranges = {k: (float(lo), float(hi)) for k, (lo, hi) in param_ranges.items()}
    ranges = _apply_floor(ranges, radius_floor)
    current = {k: 0.5 * (lo + hi) for k, (lo, hi) in ranges.items()}
    half = {k: 0.5 * (hi - lo) for k, (lo, hi) in ranges.items()}
    w, best = _fit_r2(fit_windows, features, current)
    for _ in range(levels if ranges else 0):
        for name in sorted(ranges):
            lo, hi = ranges[name]
            grid = np.clip(np.linspace(current[name] - half[name],
                                       current[name] + half[name], points),
                           lo, hi)
            for value in grid:
                trial = dict(current)
                trial[name] = float(value)
                tw, r2 = _fit_r2(fit_windows, features, trial)
                if r2 > best:
                    best, w, current = r2, tw, trial
        half = {k: v / (points - 1) for k, v in half.items()}
    Xv, yv = weak3d.equations(val_windows, features, current)
    val = _r2(Xv, yv, w)
    return Fit(weights=[float(x) for x in w], params=current, fit_r2=best,
               val_r2=val, n_terms=len(w),
               score=score(val, len(w), len(ranges)), ranges=ranges)
