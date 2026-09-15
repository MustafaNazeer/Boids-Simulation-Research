"""Attribute the weight bias to the noise that causes it.

`RESULTS-sampling-rate.md` establishes that the estimator fails by a bias
rather than by variance, that the bias is quadratic in the noise, and that
alignment carries a dt exponent one unit steeper than the other two rules,
which is what the derivation predicts because alignment is the only rule that
depends on velocity. What it leaves open is that all three exponents are about
0.55 steeper than the derivation allows.

The spec's fourth addendum registered a prediction that the textbook first
order errors in variables term,

    B^{-1} A sigma^2,  with  A = E[X_noise' y_noise] / sigma^2,  B = X_true'X_true

reproduces the measured bias. **It does not**, and by a wide margin, so this
module does not rely on it. Instead it measures the bias three ways and lets
the differences do the attribution:

- **design only**: the design matrix is built from noisy positions while the
  target is exact. Isolates everything that goes wrong inside the basis.
- **target only**: the design is exact while the target is differenced from
  noisy positions. Isolates ordinary regression noise, which should be
  unbiased.
- **both**: the real situation, where the two share the same perturbation and
  are therefore correlated.

Whatever the full bias has that the two isolated arms do not is the
correlation between them, which is the term the derivation was about. The
radii are held at their true values throughout, so this measures the weight
bias alone and needs no search.
"""

from dataclasses import dataclass

import numpy as np

from identify import basis, observe


@dataclass(frozen=True)
class Pieces:
    X_true: np.ndarray
    y_true: np.ndarray
    X_noisy: np.ndarray
    y_noisy: np.ndarray

    @property
    def X_noise(self):
        return self.X_noisy - self.X_true

    @property
    def y_noise(self):
        return self.y_noisy - self.y_true


def _design(positions, velocities, radii, dt, t0, n_frames):
    cols = []
    for t in range(t0, t0 + n_frames):
        diff, dist = basis.pairwise(positions[t])
        cols.append(np.stack([
            basis.separation(diff, dist, radii[0]),
            basis.alignment(velocities[t], dist, radii[1]),
            basis.cohesion(positions[t], dist, radii[2]),
        ], axis=-1))
    return np.concatenate(cols, axis=0).reshape(-1, 3) * dt


def truth_of(run):
    return np.array([float(run["w_separation"]),
                     float(run["w_alignment"]),
                     float(run["w_cohesion"])])


def pieces(run, sigma, seed, t0, n_frames):
    """Clean and noisy versions of the design matrix and the target."""
    P = run["positions"]
    V = run["velocities"]
    dt = float(run["dt"])
    radii = (float(run["separation_radius"]),
             float(run["alignment_radius"]),
             float(run["cohesion_radius"]))

    X_true = _design(P, V, radii, dt, t0, n_frames)
    y_true = np.concatenate(
        [V[t + 1] - V[t] for t in range(t0, t0 + n_frames)]).reshape(-1)

    P_hat = observe.add_position_noise(P, sigma, np.random.default_rng(seed))
    V_hat = observe.velocities_from_positions(P_hat, dt)
    # V_hat[k] estimates V[k+1], so the state at absolute time t uses V_hat[t-1]
    V_full = np.array(V)
    V_full[t0:t0 + n_frames] = V_hat[t0 - 1:t0 - 1 + n_frames]

    X_noisy = _design(P_hat, V_full, radii, dt, t0, n_frames)
    y_noisy = np.concatenate(
        [V_hat[t] - V_hat[t - 1] for t in range(t0, t0 + n_frames)]).reshape(-1)
    return Pieces(X_true, y_true, X_noisy, y_noisy)


def _rel_bias(X, y, truth):
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return (w - truth) / truth


def decompose(run, sigma, seeds, t0, n_frames):
    """Bias with both noises, with each one alone, and the two factors."""
    truth = truth_of(run)
    both, design_only, target_only = [], [], []
    A_sum = np.zeros(3)
    dX_sum = None
    B = None
    for seed in seeds:
        p = pieces(run, sigma, seed, t0, n_frames)
        if B is None:
            B = p.X_true.T @ p.X_true
            dX_sum = np.zeros_like(p.X_true)
        both.append(_rel_bias(p.X_noisy, p.y_noisy, truth))
        design_only.append(_rel_bias(p.X_noisy, p.y_true, truth))
        target_only.append(_rel_bias(p.X_true, p.y_noisy, truth))
        A_sum += p.X_noise.T @ p.y_noise
        dX_sum += p.X_noise
    n = len(seeds)
    A = A_sum / n / (sigma ** 2) if sigma > 0 else np.zeros(3)
    both = np.array(both).mean(axis=0)
    design_only = np.array(design_only).mean(axis=0)
    target_only = np.array(target_only).mean(axis=0)
    return {
        "dt": float(run["dt"]),
        "sigma": sigma,
        "n_seeds": n,
        "both": both,
        "design_only": design_only,
        "target_only": target_only,
        # what the full bias has that neither isolated arm supplies
        "correlation": both - design_only - target_only,
        "A": A,
        "B": B,
        "first_order": np.linalg.solve(B, A) * (sigma ** 2) / truth,
        # the mean shift of the design matrix, which is nonzero because the
        # basis is curved and E[g(p + e)] is therefore not g(p)
        "mean_design_shift": float(np.abs(dX_sum / n).mean()),
    }
