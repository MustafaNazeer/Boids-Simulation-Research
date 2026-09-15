"""Write a controlled sweep manifest without touching the simulator.

`sweep.py` splits planning from execution: `plan` draws parameters and writes
`manifest.csv`, `run` reads that file and executes the rows it is given. Every
field `run` consumes is a manifest column, including `frames`, `dt`,
`world_size` and `bounds`, so a controlled sweep needs only a different
planner. Nothing under `sim/` changes.

`sweep plan` cannot be used here because `collect.sample_params` draws the
predator, the obstacles, the leaders, the preplanned agents and the food
sources at random, and the three term estimator models none of them. Of the
1000 runs in the August `local_sweep`, exactly one has all five at zero.

Protocol 1 of the sampling rate experiment compares timesteps, so the same
physical system must appear at every timestep. Parameters are therefore drawn
once per run index and reused across the timesteps, which
`test_the_same_physics_is_shared_across_timesteps` enforces.
"""

import numpy as np

from sim.boids import collect

N_BOIDS = 80
MAX_SPEED = 8.0
WORLD_SIZE = 60.0
BOUNDS = "none"

# the sampled keys, and the only ones that vary between runs
_SAMPLED = ("w_separation", "w_alignment", "w_cohesion",
            "separation_radius", "alignment_radius", "cohesion_radius")


def _draw(rng):
    out = {}
    for key in _SAMPLED:
        lo, hi = collect.PARAM_RANGES[key]
        out[key] = float(rng.uniform(lo, hi))
    out["seed"] = int(rng.integers(0, 2 ** 31 - 1))
    return out


def plan(dts, runs_per_dt, frames_for, seed):
    """Rows for a controlled sweep, one group per timestep.

    `frames_for` maps each timestep to its frame count, so the caller decides
    whether the arms are matched on frame count or on simulated duration.
    """
    rng = np.random.default_rng(seed)
    systems = [_draw(rng) for _ in range(runs_per_dt)]
    rows = []
    run_id = 0
    for dt in dts:
        for system in systems:
            row = {
                "run_id": run_id,
                "seed": system["seed"],
                "n_boids": N_BOIDS,
                "frames": int(frames_for[dt]),
                "dt": float(dt),
                "world_size": WORLD_SIZE,
                "max_speed": MAX_SPEED,
                # every extra behaviour off, which is the whole point
                "predator": 0,
                "obstacles": 0,
                "n_preplanned": 0,
                "n_leaders": 0,
                "n_food": 0,
                # unused, but the header requires them and row_to_cfg reads
                # them, so they carry the simulator's own defaults
                "leader_follow_radius": 20.0,
                "w_follow": 1.0,
                "w_wander": 2.0,
                "food_sensing_radius": 20.0,
                "w_food_seek": 1.0,
                "bounds": BOUNDS,
            }
            row.update({k: system[k] for k in _SAMPLED})
            rows.append({k: row[k] for k in collect.MANIFEST_HEADER})
            run_id += 1
    return rows


def write(rows, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    collect.write_manifest(rows, os.path.join(out_dir, "manifest.csv"))
    return os.path.join(out_dir, "manifest.csv")
