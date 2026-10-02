"""The two ground truth systems, written as candidate sources.

They use exactly the format the LLM must produce, so the same text can be
fed through the sandbox and the verifier, and a pass on these is a pass of
the whole pipeline rather than of a side path. The boids terms are the
definitions in meetings/september/28/identify/basis.py, which work in any
dimension.
"""

import numpy as np

BOIDS = '''
import numpy as np

PARAMS = {"r_sep": (2.0, 12.0), "r_ali": (5.0, 25.0), "r_coh": (5.0, 25.0)}


def features(pos, vel, center, params):
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    m = (dist < params["r_sep"]) & (dist > 0.0)
    safe = np.where(m, dist, 1.0)
    n = m.sum(axis=1)
    sep = np.einsum("ij,ijk->ik", np.where(m, 1.0 / (safe * safe), 0.0), diff)
    sep = sep / np.maximum(n, 1)[:, None]
    m = dist < params["r_ali"]
    n = m.sum(axis=1)
    ali = (m.astype(float) @ vel) / np.maximum(n, 1)[:, None] - vel
    ali[n == 0] = 0.0
    m = dist < params["r_coh"]
    n = m.sum(axis=1)
    coh = (m.astype(float) @ pos) / np.maximum(n, 1)[:, None] - pos
    coh[n == 0] = 0.0
    return np.stack([sep, ali, coh], axis=-1)
'''

WELL = '''
import numpy as np

PARAMS = {"r_rep": (1.0, 10.0)}


def features(pos, vel, center, params):
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    kernel = np.exp(-(dist / params["r_rep"]) ** 2)
    rep = np.einsum("ij,ijk->ik", kernel, diff)
    return np.stack([center - pos, rep, -vel], axis=-1)
'''

BOIDS_PARAMS = {"r_sep": 7.0, "r_ali": 16.0, "r_coh": 16.0}
BOIDS_WEIGHTS = np.array([8.0, 1.6, 0.5])
WELL_PARAMS = {"r_rep": 3.0}
WELL_WEIGHTS = np.array([0.5, 20.0, 0.8])


def features_of(source):
    """Load a trusted, in repo source. Never call this on LLM output."""
    namespace = {}
    exec(compile(source, "<truth>", "exec"), namespace)
    return namespace["features"]


def simulate(source, params, weights, n_agents, frames, dt, seed,
             kick=0.0, spread=10.0, speed=1.0):
    """Integrate p[t+1] = 2 p[t] - p[t-1] + dt^2 a[t] exactly.

    v[t] = (p[t] - p[t-1]) / dt is what the features see and what is stored,
    so the weak form identity holds to rounding when kick is zero.
    """
    rng = np.random.default_rng(seed)
    feats = features_of(source)
    w = np.asarray(weights, dtype=float)
    P = np.empty((frames, n_agents, 3))
    V = np.empty_like(P)
    P[0] = rng.normal(scale=spread, size=(n_agents, 3))
    V[0] = rng.normal(scale=speed, size=(n_agents, 3))
    P[1] = P[0] + dt * V[0]
    V[1] = (P[1] - P[0]) / dt
    for t in range(1, frames - 1):
        a = feats(P[t], V[t], P[t].mean(axis=0), params) @ w
        if kick:
            a = a + rng.normal(scale=kick, size=a.shape)
        P[t + 1] = 2.0 * P[t] - P[t - 1] + dt * dt * a
        V[t + 1] = (P[t + 1] - P[t]) / dt
    return P, V
