"""The three classic rules as design matrix columns.

`sim.boids.rules` computes each rule with a KD tree and a Python loop over
neighbours, which is the right shape for running a simulation and the wrong
shape for evaluating the same rule at several hundred candidate radii on the
same frame. This module recomputes them from a cached pairwise distance
matrix instead, so a whole radius scan reuses one distance computation.

It deliberately does not import `sim`. The equivalence with `sim.boids.rules`
is asserted in `identify/tests/test_basis.py` on both random and recorded
frames, in the same spirit as `sim/tests/test_frozen_collect.py`, which proves
the two collectors agree rather than trusting that they do.
"""

import numpy as np


def pairwise(positions):
    """Return the offset tensor p_i - p_j and the distance matrix.

    The diagonal of the distance matrix is set to infinity so that an agent is
    never its own neighbour under any finite radius.
    """
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return diff, dist


def separation(diff, dist, radius):
    # rules.separation skips a coincident pair entirely and does not count it
    # toward the mean, so the mask carries dist > 0 as well as dist < radius
    mask = (dist < radius) & (dist > 0.0)
    safe = np.where(mask, dist, 1.0)
    weight = np.where(mask, 1.0 / (safe * safe), 0.0)
    count = mask.sum(axis=1)
    out = np.einsum("ij,ijk->ik", weight, diff)
    out /= np.maximum(count, 1)[:, None]
    out[count == 0] = 0.0
    return out


def alignment(velocities, dist, radius):
    mask = dist < radius
    count = mask.sum(axis=1)
    out = (mask.astype(float) @ velocities) / np.maximum(count, 1)[:, None]
    out -= velocities
    out[count == 0] = 0.0
    return out


def cohesion(positions, dist, radius):
    mask = dist < radius
    count = mask.sum(axis=1)
    out = (mask.astype(float) @ positions) / np.maximum(count, 1)[:, None]
    out -= positions
    out[count == 0] = 0.0
    return out


def frame_bases(positions, velocities, sep_radius, ali_radius, coh_radius):
    """All three rule accelerations for one frame, as (N, 2) arrays."""
    diff, dist = pairwise(positions)
    return (separation(diff, dist, sep_radius),
            alignment(velocities, dist, ali_radius),
            cohesion(positions, dist, coh_radius))
