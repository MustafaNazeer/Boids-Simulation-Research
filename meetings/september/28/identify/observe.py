"""The observation model: noisy positions in, regression windows out.

A tracker records positions, not velocities, so the estimator is given
positions and must differentiate them. That single fact is what the sampling
rate experiment is about, and getting its index arithmetic wrong would make
every downstream number meaningless, so the mapping is written out once here
and checked in `identify/tests/test_observe.py`.

The simulator integrates as

    v[t+1] = clamp(v[t] + a(p[t], v[t]) * dt)
    p[t+1] = bounds(p[t] + v[t+1] * dt)

so in the unbounded world the first difference of positions is not an
approximation of the velocity, it is the velocity, exactly:

    (p[t+1] - p[t]) / dt == v[t+1]

Writing d[k] for that first difference, d[k] == v[k+1], so the state at
absolute time t is (p[t], d[t-1]) and the regression target at time t is
d[t] - d[t-1]. Both need t >= 1, and the target needs t <= T - 2.
"""

from dataclasses import dataclass

import numpy as np

from identify import basis


def add_position_noise(positions, sigma, rng):
    """Additive isotropic Gaussian noise. A sigma of zero is the identity."""
    if sigma == 0.0:
        return positions
    return positions + rng.normal(0.0, sigma, size=positions.shape)


def velocities_from_positions(positions, dt):
    """First differences. Entry k estimates the velocity at time k + 1."""
    return (positions[1:] - positions[:-1]) / dt


@dataclass(frozen=True)
class Frame:
    """One frame of observed state, with its pairwise geometry cached.

    The offsets and distances are computed once per frame and reused across
    every candidate radius, which is what makes a radius scan affordable.
    """
    positions: np.ndarray
    velocities: np.ndarray
    diff: np.ndarray
    dist: np.ndarray


@dataclass(frozen=True)
class Observation:
    positions: np.ndarray
    diffs: np.ndarray
    dt: float

    @classmethod
    def from_positions(cls, positions, dt):
        return cls(positions=positions,
                   diffs=velocities_from_positions(positions, dt),
                   dt=float(dt))

    def max_window_start_plus_frames(self):
        """The largest permitted value of t0 + n_frames."""
        return self.positions.shape[0] - 1

    def span(self, t0, n_frames, stride=1):
        """How many frames of the run a window touches, first anchor to last."""
        return (n_frames - 1) * stride + 1

    def window(self, t0, n_frames, stride=1):
        """Frames and regression target at times t0, t0+stride, ...

        `stride` thins the anchors without changing anything else. Each sample
        still differences consecutive frames for its own target, so spreading
        the anchors out introduces no discretisation bias; it only changes
        which moments of the trajectory are looked at. That is what separates
        "more frames" from "a longer span", which the two protocols of the
        sampling rate experiment confound.
        """
        if t0 < 1:
            raise ValueError("t0 must be at least 1: the state at time t "
                             "needs the first difference ending at t")
        if stride < 1:
            raise ValueError("stride must be at least 1, got %r" % (stride,))
        limit = self.max_window_start_plus_frames()
        last = t0 + (n_frames - 1) * stride
        if last + 1 > limit:
            raise ValueError(
                "window t0=%d n_frames=%d stride=%d needs frame %d, past the "
                "usable end %d" % (t0, n_frames, stride, last + 1, limit))
        frames = []
        targets = []
        for t in range(t0, last + 1, stride):
            p = self.positions[t]
            v = self.diffs[t - 1]
            diff, dist = basis.pairwise(p)
            frames.append(Frame(p, v, diff, dist))
            targets.append(self.diffs[t] - self.diffs[t - 1])
        return frames, np.concatenate(targets)


def observed(positions, dt, sigma, rng):
    """Convenience: add noise to positions and build the Observation."""
    return Observation.from_positions(
        add_position_noise(positions, sigma, rng), dt)
