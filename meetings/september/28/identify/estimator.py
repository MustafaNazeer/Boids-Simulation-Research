"""Profile least squares recovery of the three radii and the three weights.

The structure that makes this cheap is that the acceleration is linear in the
weights once the radii are fixed:

    a_i = w_sep * sep_i(R_sep) + w_ali * ali_i(R_ali) + w_coh * coh_i(R_coh)

So the weights are profiled out in closed form by `numpy.linalg.lstsq` at every
candidate radius triple, and only the three radii are searched. The search is
a deterministic coarse to fine coordinate descent; there is no random restart
and no learning rate, so the same inputs always give the same answer, which
`test_estimate_is_deterministic` enforces.

The radii enter only through which pairs fall inside them, so the residual is
piecewise smooth in each radius with kinks where a pair crosses. A grid search
is therefore the honest method: a gradient would be zero almost everywhere.
"""

from dataclasses import dataclass

import numpy as np

from identify import basis

# Search spans, deliberately wider than the generating ranges in
# sim.boids.collect.PARAM_RANGES (5 to 9 and 12 to 20) so the estimator is not
# boxed into the right answer by its own search bounds.
SPANS = ((4.0, 10.0), (10.0, 22.0), (10.0, 22.0))

# Coarse to fine step sizes, in world units. The last one sets the resolution
# floor on any recovered radius and must be reported beside any error smaller
# than itself.
LEVELS = (0.5, 0.1, 0.02)

_PASSES_PER_LEVEL = 2
_HALF_WIDTH = 12


@dataclass(frozen=True)
class Fit:
    radii: np.ndarray
    weights: np.ndarray
    rmse: float


def design(frames, radii):
    """Stack the three rule accelerations into an (M, 2, 3) design tensor."""
    sep_r, ali_r, coh_r = radii
    cols = []
    for fr in frames:
        cols.append(np.stack([
            basis.separation(fr.diff, fr.dist, sep_r),
            basis.alignment(fr.velocities, fr.dist, ali_r),
            basis.cohesion(fr.positions, fr.dist, coh_r),
        ], axis=-1))
    return np.concatenate(cols, axis=0)


def profile(frames, target, keep, dt, radii):
    """Least squares weights and residual RMSE at one radius triple."""
    X = design(frames, radii)[keep].reshape(-1, 3) * dt
    y = target[keep].reshape(-1)
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ w
    return Fit(radii=np.asarray(radii, dtype=float), weights=w,
               rmse=float(np.sqrt(np.mean(residual * residual))))


def estimate(frames, target, keep, dt, spans=SPANS, levels=LEVELS):
    """Coarse to fine coordinate descent over the three radii."""
    # ties are broken by the ascending scan keeping the first minimum, so on a
    # flat set this returns its lowest grid point. That is deterministic but it
    # is also a systematic downward bias, which is the other reason a point
    # estimate alone is not an honest report. See identified_interval.
    current = [0.5 * (lo + hi) for lo, hi in spans]
    for step in levels:
        for _ in range(_PASSES_PER_LEVEL):
            for axis in range(3):
                lo, hi = spans[axis]
                centre = current[axis]
                grid = np.arange(max(lo, centre - _HALF_WIDTH * step),
                                 min(hi, centre + _HALF_WIDTH * step) + 1e-9,
                                 step)
                best_value = np.inf
                best_x = centre
                for x in grid:
                    trial = list(current)
                    trial[axis] = float(x)
                    value = profile(frames, target, keep, dt, trial).rmse
                    if value < best_value:
                        best_value = value
                        best_x = float(x)
                current[axis] = best_x
    return profile(frames, target, keep, dt, current)


def separation_only_radius(frames, target, keep, dt, span=SPANS[0], step=0.05):
    """The trivial competitor: fit the separation term alone.

    `RESULTS.md` caught this project reporting 0.999 AUC beside a trivial
    method that scored the same. The equivalent trap here is a three rule fit
    that looks good only because one rule already explains the data, so this
    reference is computed for every arm and reported beside the full fit
    rather than discovered afterwards.
    """
    y = target[keep].reshape(-1)
    best_value = np.inf
    best_x = span[0]
    for x in np.arange(span[0], span[1] + 1e-9, step):
        cols = [basis.separation(fr.diff, fr.dist, float(x)) for fr in frames]
        X = np.concatenate(cols, axis=0)[keep].reshape(-1, 1) * dt
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ w
        value = float(np.sqrt(np.mean(residual * residual)))
        if value < best_value:
            best_value = value
            best_x = float(x)
    return best_x


def identified_interval(frames, radius):
    """The set of radii the data cannot tell apart from `radius`.

    A radius enters the model only through the mask `dist < radius`, so two
    radii are indistinguishable exactly when no pair distance separates them.
    Writing `lo` for the largest observed distance below `radius` and `hi` for
    the smallest at or above it, every r in (lo, hi] produces the identical
    design matrix and therefore the identical residual, to the last bit.

    This is why a recovered radius is reported as an interval and not as a
    point. Measured on the 24 agent 8 frame test fixture, the interval is
    about 0.1 wide, which is five times the 0.02 search grid, so a point
    estimate there would be quoting precision the data does not contain.
    """
    lo = 0.0
    hi = np.inf
    for fr in frames:
        finite = fr.dist[np.isfinite(fr.dist)]
        below = finite[finite < radius]
        above = finite[finite >= radius]
        if below.size:
            lo = max(lo, float(below.max()))
        if above.size:
            hi = min(hi, float(above.min()))
    return lo, hi


def within_identified_interval(frames, estimated, truth):
    """Is the estimate indistinguishable from the truth, given this data."""
    lo, hi = identified_interval(frames, truth)
    return lo < estimated <= hi


def unclamped_mask(velocities, max_speed, min_speed, t0, n_frames, stride=1,
                   tol=1e-9):
    """Agent frames whose recorded velocity was not touched by a speed limit.

    Clamping is a hard nonlinearity and breaks the linear model on whatever
    agent frames it touches. Detecting it exactly needs the true speeds, so
    this is an oracle mask, and every results file using it must say so.
    """
    last = t0 + (n_frames - 1) * stride
    speeds = np.linalg.norm(velocities[t0:last + 1:stride], axis=2)
    clamped = np.abs(speeds - max_speed) < tol
    if min_speed > 0.0:
        clamped |= np.abs(speeds - min_speed) < tol
    return (~clamped).reshape(-1)


def relative_error(estimated, truth):
    return np.abs(np.asarray(estimated) - np.asarray(truth)) / np.abs(truth)
