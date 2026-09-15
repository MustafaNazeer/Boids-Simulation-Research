"""Three assertions that must pass before any number is emitted.

Their purpose is to make a null result interpretable. Without them, "the
effect is not there" and "the estimator is broken" are the same observation,
and this project has already spent a results file explaining that a 0.998
score measured nothing except a leak.

1. forward model identity: our reimplementation of the step reproduces the
   recorded trajectory. If it does not, the observation pipeline and the
   simulator have diverged and nothing downstream means anything.
2. zero noise exactness: with clean data at the true radii, the residual is
   at machine precision. This is what licenses reading a later failure as a
   statement about information rather than about the method.
3. null calibration: handed a different system's trajectory, the estimator
   reports that other system's parameters and is therefore wrong about the
   target by roughly the prior width. An estimator that is right anyway is
   reading something it should not be.
"""

from dataclasses import dataclass

import numpy as np

from identify import basis, estimator, observe

NULL_BAND = (0.10, 0.25)


class TripwireFailure(AssertionError):
    pass


@dataclass(frozen=True)
class TripwireResult:
    name: str
    passed: bool
    value: float
    detail: str


def _f(run, key):
    return float(run[key])


def _apply_bounds(positions, velocities, size, mode):
    if mode == "wrap":
        return np.mod(positions, size), velocities
    if mode == "none":
        return positions, velocities
    pos = positions.copy()
    vel = velocities.copy()
    low = pos < 0.0
    pos[low] = -pos[low]
    vel[low] = np.abs(vel[low])
    high = pos > size
    pos[high] = 2.0 * size - pos[high]
    vel[high] = -np.abs(vel[high])
    return pos, vel


def _clamp(velocities, max_speed, min_speed):
    out = velocities.copy()
    speeds = np.linalg.norm(out, axis=1)
    fast = speeds > max_speed
    out[fast] = out[fast] / speeds[fast, None] * max_speed
    speeds = np.linalg.norm(out, axis=1)
    slow = (speeds < min_speed) & (speeds > 0)
    out[slow] = out[slow] / speeds[slow, None] * min_speed
    return out


def _bounds_mode(run):
    mode = run["bounds"]
    return mode.item() if hasattr(mode, "item") else str(mode)


def forward_model_identity(run, min_speed=0.0, tol=1e-12):
    """Reproduce every recorded transition from the stored parameters.

    `min_speed` is a parameter rather than a stored field because the npz does
    not carry it: `collect.build_world` derives it as
    min(config min_speed, max_speed / 2) and never writes it out. Data
    generated with `identify/threerule.yaml` has it at zero.
    """
    P = run["positions"]
    V = run["velocities"]
    dt = _f(run, "dt")
    size = _f(run, "world_size")
    mode = _bounds_mode(run)
    ws, wa, wc = (_f(run, "w_separation"), _f(run, "w_alignment"),
                  _f(run, "w_cohesion"))
    radii = (_f(run, "separation_radius"), _f(run, "alignment_radius"),
             _f(run, "cohesion_radius"))
    max_speed = _f(run, "max_speed")

    worst = 0.0
    for t in range(P.shape[0] - 1):
        sep, ali, coh = basis.frame_bases(P[t], V[t], *radii)
        v = V[t] + (ws * sep + wa * ali + wc * coh) * dt
        v = _clamp(v, max_speed, min_speed)
        p, v = _apply_bounds(P[t] + v * dt, v, size, mode)
        worst = max(worst,
                    float(np.abs(v - V[t + 1]).max()),
                    float(np.abs(p - P[t + 1]).max()))
    return TripwireResult(
        "forward_model_identity", worst < tol, worst,
        "max absolute state error over %d transitions is %.3e (tol %.0e)"
        % (P.shape[0] - 1, worst, tol))


def _clean_window(run, n_frames, sigma, seed):
    P = run["positions"]
    dt = _f(run, "dt")
    rng = np.random.default_rng(seed)
    obs = observe.observed(P, dt, sigma, rng)
    frames, target = obs.window(t0=1, n_frames=n_frames)
    keep = estimator.unclamped_mask(run["velocities"], _f(run, "max_speed"),
                                    0.0, t0=2, n_frames=n_frames)
    return frames, target, keep, dt


def zero_noise_exactness(run, n_frames=8, sigma=0.0, seed=0, tol=1e-12):
    frames, target, keep, dt = _clean_window(run, n_frames, sigma, seed)
    radii = (_f(run, "separation_radius"), _f(run, "alignment_radius"),
             _f(run, "cohesion_radius"))
    fit = estimator.profile(frames, target, keep, dt, radii)
    truth = np.array([_f(run, "w_separation"), _f(run, "w_alignment"),
                      _f(run, "w_cohesion")])
    weight_error = float(estimator.relative_error(fit.weights, truth).max())
    passed = fit.rmse < tol and weight_error < 1e-6
    return TripwireResult(
        "zero_noise_exactness", passed, fit.rmse,
        "residual rmse %.3e (tol %.0e), max relative weight error %.3e"
        % (fit.rmse, tol, weight_error))


def null_calibration(target_run, data_run, n_frames=8, band=NULL_BAND):
    """Fit on one system's data, score against another system's truth."""
    frames, target, keep, dt = _clean_window(data_run, n_frames, 0.0, 0)
    fit = estimator.estimate(frames, target, keep, dt)
    truth = np.array([_f(target_run, "separation_radius"),
                      _f(target_run, "alignment_radius"),
                      _f(target_run, "cohesion_radius")])
    error = float(estimator.relative_error(fit.radii, truth).mean())
    passed = band[0] <= error <= band[1]
    return TripwireResult(
        "null_calibration", passed, error,
        "mean relative radius error against the wrong truth is %.4f, "
        "expected inside [%.2f, %.2f]" % (error, band[0], band[1]))


def run_all(run, other_run, n_frames=8, min_speed=0.0):
    return {
        "forward_model_identity": forward_model_identity(run, min_speed),
        "zero_noise_exactness": zero_noise_exactness(run, n_frames),
        "null_calibration": null_calibration(run, other_run, n_frames),
    }


def require(run, other_run, n_frames=8, min_speed=0.0):
    """Raise unless every tripwire is green. No result is emitted otherwise."""
    results = run_all(run, other_run, n_frames, min_speed)
    red = [r for r in results.values() if not r.passed]
    if red:
        raise TripwireFailure(
            "tripwires failed, refusing to emit results:\n" +
            "\n".join("  %s: %s" % (r.name, r.detail) for r in red))
    return results
