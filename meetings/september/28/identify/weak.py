"""The integral, or weak, formulation: never differentiate the data.

`docs/literature-notes.md` Part 3 records that this is the standard remedy in
the equation discovery literature for the failure `RESULTS-sampling-rate.md`
measures, and that we had not tested it.

The simulator's discrete relation is exact:

    (p[t+1] - 2 p[t] + p[t-1]) / dt^2  =  a_t  =  sum_r w_r g_r(p_t, v_t)

Multiplying by a test function that vanishes with its first difference at both
ends of a window, and summing by parts, moves both differences onto the test
function:

    sum_t (D2 phi)_t p_t / dt^2  =  sum_t phi_t sum_r w_r g_r(p_t, v_t)

The left side is then a fixed smooth weighted sum of positions. Noise in it
averages down rather than amplifying, which is the opposite of what a second
difference does.

**What this does and does not fix**, per the spec's fifth addendum. It does
not remove the dt scaling: the relation carries dt^2 whatever test function is
used, because the acceleration does not depend on dt and the positional
combination does. What it removes is the target noise, and `identify/eiv.py`
measured that the observed bias is the difference of an inflating correlation
term, which needs target noise, and an attenuating design term, which does
not. So the prediction is a sign flip rather than a collapse.

It also fixes the three rules unequally. Separation and cohesion depend only
on positions, so under the weak form no differentiation survives anywhere in
their equations. Alignment depends on velocity, so its design column still
needs differenced velocities.
"""

from dataclasses import dataclass

import numpy as np

from identify import basis, observe


def test_function(n):
    """A smooth bump vanishing with its first difference at both ends.

    A squared Hann window. Squaring is what makes the first difference vanish
    at the ends as well as the value, which is what kills the boundary term in
    the summation by parts.
    """
    t = np.arange(n)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / (n - 1)))
    return hann ** 2


def second_difference(phi):
    """The discrete second difference, padded so it aligns with phi itself."""
    out = np.zeros_like(phi)
    out[1:-1] = phi[2:] - 2.0 * phi[1:-1] + phi[:-2]
    # phi and its first difference vanish at the ends, so the two edge entries
    # of the second difference are the only places a boundary term could hide.
    # They are written out rather than dropped so the identity stays exact.
    out[0] = phi[1] - 2.0 * phi[0]
    out[-1] = phi[-2] - 2.0 * phi[-1]
    return out


@dataclass(frozen=True)
class Equations:
    X: np.ndarray
    y: np.ndarray
    n_windows: int


def _frames(positions, velocities, radii, t0, n_frames):
    cols = []
    for t in range(t0, t0 + n_frames):
        diff, dist = basis.pairwise(positions[t])
        cols.append(np.stack([
            basis.separation(diff, dist, radii[0]),
            basis.alignment(velocities[t], dist, radii[1]),
            basis.cohesion(positions[t], dist, radii[2]),
        ], axis=-1))
    return np.stack(cols)          # (n_frames, agents, 2, 3)


def equations(run, sigma, seed, t0, n_frames, window, radii=None):
    """Weak form equations over every overlapping window of the given length.

    One equation per agent per spatial component per window. Windows overlap
    at stride one, so a 50 frame stretch with a 20 frame window still supplies
    31 windows and the equation count stays comparable to the pointwise form.
    """
    if window > n_frames:
        raise ValueError("window %d does not fit in %d frames"
                         % (window, n_frames))
    if window < 5:
        raise ValueError("window must be at least 5 frames")

    P = run["positions"]
    dt = float(run["dt"])
    if radii is None:
        radii = (float(run["separation_radius"]),
                 float(run["alignment_radius"]),
                 float(run["cohesion_radius"]))

    P_hat = observe.add_position_noise(P, sigma, np.random.default_rng(seed))
    V_hat = observe.velocities_from_positions(P_hat, dt)
    V_full = np.array(run["velocities"])
    V_full[t0:t0 + n_frames] = V_hat[t0 - 1:t0 - 1 + n_frames]

    G = _frames(P_hat, V_full, radii, t0, n_frames)
    phi = test_function(window)
    d2 = second_difference(phi)

    ys, Xs = [], []
    for s in range(n_frames - window + 1):
        seg = slice(t0 + s, t0 + s + window)
        # target: the smooth weighted sum of positions, no differencing
        ys.append(np.einsum("t,tac->ac", d2, P_hat[seg]) / (dt * dt))
        Xs.append(np.einsum("t,tacr->acr", phi, G[s:s + window]))
    y = np.concatenate([a.reshape(-1) for a in ys])
    X = np.concatenate([a.reshape(-1, 3) for a in Xs])
    return Equations(X=X, y=y, n_windows=len(ys))


def target_noise_scale(run, window):
    """Standard deviation of the weak target per unit sigma, per equation."""
    dt = float(run["dt"])
    d2 = second_difference(test_function(window))
    return float(np.sqrt(np.sum(d2 * d2)) / (dt * dt))


def pointwise_target_noise_scale(run):
    """The same quantity for a plain second difference: sqrt(6) / dt^2."""
    dt = float(run["dt"])
    return float(np.sqrt(6.0) / (dt * dt))


def snr_gain(run, window):
    """How much better the weak target's signal to noise is, per equation.

    Comparing raw target noise alone understates the method, because the weak
    equation also accumulates signal: its right hand side sums the
    acceleration against the test function over the whole window. The honest
    comparison is the ratio of the two signal to noise ratios,

        (sum phi) / ||D2 phi||   against   1 / sqrt(6)

    with the dt^2 cancelling, which is the algebraic reason the weak form
    cannot change the dt scaling however large this gain is.
    """
    phi = test_function(window)
    d2 = second_difference(phi)
    return float(phi.sum() / np.sqrt(np.sum(d2 * d2)) * np.sqrt(6.0))


def bias(run, sigma, seeds, t0, n_frames, window):
    """Relative weight bias under the weak form, radii held at truth."""
    truth = np.array([float(run["w_separation"]),
                      float(run["w_alignment"]),
                      float(run["w_cohesion"])])
    out = []
    for seed in seeds:
        e = equations(run, sigma, seed, t0, n_frames, window)
        w, _, _, _ = np.linalg.lstsq(e.X, e.y, rcond=None)
        out.append((w - truth) / truth)
    return np.array(out).mean(axis=0)


def profile(run, sigma, seed, t0, n_frames, window, radii):
    """Weak form residual and weights at one candidate radius triple."""
    e = equations(run, sigma, seed, t0, n_frames, window, radii=radii)
    w, _, _, _ = np.linalg.lstsq(e.X, e.y, rcond=None)
    r = e.y - e.X @ w
    return float(np.sqrt(np.mean(r * r))), w


def estimate(run, sigma, seed, t0, n_frames, window,
             spans=((4.0, 10.0), (10.0, 22.0), (10.0, 22.0)),
             levels=(0.5, 0.1, 0.02)):
    """Coarse to fine coordinate descent over the radii, weak form residual.

    The same deterministic search `identify/estimator.estimate` uses, so the
    two are comparable arm for arm; only the equations underneath differ.
    """
    current = [0.5 * (lo + hi) for lo, hi in spans]
    for step in levels:
        for _ in range(2):
            for axis in range(3):
                lo, hi = spans[axis]
                c = current[axis]
                grid = np.arange(max(lo, c - 12 * step),
                                 min(hi, c + 12 * step) + 1e-9, step)
                best = (np.inf, c)
                for x in grid:
                    trial = list(current)
                    trial[axis] = float(x)
                    v, _ = profile(run, sigma, seed, t0, n_frames, window,
                                   trial)
                    if v < best[0]:
                        best = (v, float(x))
                current[axis] = best[1]
    _, w = profile(run, sigma, seed, t0, n_frames, window, current)
    truth = np.array([float(run["w_separation"]),
                      float(run["w_alignment"]),
                      float(run["w_cohesion"])])
    true_r = np.array([float(run["separation_radius"]),
                       float(run["alignment_radius"]),
                       float(run["cohesion_radius"])])
    return {
        "radii": np.array(current),
        "weights": w,
        "weight_rel_err": np.abs(w - truth) / np.abs(truth),
        "radius_rel_err": np.abs(np.array(current) - true_r) / true_r,
    }
