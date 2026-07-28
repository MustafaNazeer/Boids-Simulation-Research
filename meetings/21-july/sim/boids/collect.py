import os
import csv
import copy
import argparse
import numpy as np

from sim.boids.world import World, default_params
from sim.boids import config as config_mod
from sim.boids import preplanned
from sim.boids.leader import follow as leader_follow
from sim.boids.recorder import TrajectoryRecorder

PARAM_RANGES = {
    "n_boids": (40, 160),
    "w_separation": (4.0, 10.0),
    "w_alignment": (0.8, 2.5),
    "w_cohesion": (0.3, 1.0),
    "separation_radius": (5.0, 9.0),
    "alignment_radius": (12.0, 20.0),
    "cohesion_radius": (12.0, 20.0),
    "max_speed": (3.0, 7.0),
    "n_preplanned": (0, 4),
    "n_leaders": (0, 5),
    "leader_follow_radius": (20.0, 40.0),
    "w_follow": (0.4, 2.0),
    "w_wander": (1.0, 3.0),
}

MANIFEST_HEADER = [
    "run_id", "seed", "n_boids", "frames", "dt", "world_size",
    "w_separation", "w_alignment", "w_cohesion",
    "separation_radius", "alignment_radius", "cohesion_radius",
    "max_speed", "predator", "obstacles", "n_preplanned",
    "n_leaders", "leader_follow_radius", "w_follow", "w_wander",
]


def _uniform(rng, key):
    lo, hi = PARAM_RANGES[key]
    return float(rng.uniform(lo, hi))


def sample_params(rng):
    nlo, nhi = PARAM_RANGES["n_boids"]
    plo, phi = PARAM_RANGES["n_preplanned"]
    llo, lhi = PARAM_RANGES["n_leaders"]
    return {
        "n_boids": int(rng.integers(nlo, nhi + 1)),
        "weights": {
            "separation": _uniform(rng, "w_separation"),
            "alignment": _uniform(rng, "w_alignment"),
            "cohesion": _uniform(rng, "w_cohesion"),
        },
        "separation_radius": _uniform(rng, "separation_radius"),
        "alignment_radius": _uniform(rng, "alignment_radius"),
        "cohesion_radius": _uniform(rng, "cohesion_radius"),
        "max_speed": _uniform(rng, "max_speed"),
        "predator": bool(rng.random() < 0.5),
        "obstacles": bool(rng.random() < 0.5),
        "n_preplanned": int(rng.integers(plo, phi + 1)),
        "n_leaders": int(rng.integers(llo, lhi + 1)),
        "leader_follow_radius": _uniform(rng, "leader_follow_radius"),
        "w_follow": _uniform(rng, "w_follow"),
        "w_wander": _uniform(rng, "w_wander"),
    }


def build_world(cfg, run_seed, frames, world_size, base=None):
    rng = np.random.default_rng(run_seed)
    n = cfg["n_boids"]
    positions = rng.uniform(0, world_size, size=(n, 2))
    velocities = rng.uniform(-1, 1, size=(n, 2))

    # the config supplies the fixed settings, above all which rules are on;
    # the sweep then varies only the numeric parameters it owns
    params = default_params() if base is None else copy.deepcopy(base)
    params["leader_follow_radius"] = cfg["leader_follow_radius"]
    params["weights"]["follow"] = cfg["w_follow"]
    params["weights"]["wander"] = cfg["w_wander"]
    params["separation_radius"] = cfg["separation_radius"]
    params["alignment_radius"] = cfg["alignment_radius"]
    params["cohesion_radius"] = cfg["cohesion_radius"]
    params["max_speed"] = cfg["max_speed"]
    params["min_speed"] = min(params["min_speed"], cfg["max_speed"] * 0.5)
    params["preplanned_speed"] = cfg["max_speed"]
    params["weights"]["separation"] = cfg["weights"]["separation"]
    params["weights"]["alignment"] = cfg["weights"]["alignment"]
    params["weights"]["cohesion"] = cfg["weights"]["cohesion"]

    obstacles = None
    if cfg["obstacles"]:
        centers = np.array([[world_size * 0.30, world_size * 0.63],
                            [world_size * 0.73, world_size * 0.33]])
        radii = np.array([world_size * 0.08, world_size * 0.08])
        obstacles = (centers, radii)

    predator_state = None
    if cfg["predator"]:
        predator_state = {"pos": np.array([world_size * 0.92, world_size * 0.92])}

    mask = np.zeros(n, dtype=bool)
    agents = None
    n_preplanned = cfg["n_preplanned"]
    if n_preplanned > 0:
        pre_ids = np.sort(rng.choice(n, size=n_preplanned, replace=False))
        mask[pre_ids] = True
        plans = preplanned.PLAN_CATALOG[:n_preplanned]
        center = np.array([world_size / 2.0, world_size / 2.0])
        amplitude = (world_size / 2.0) * 0.8
        agents = preplanned.PreplannedAgents(
            plans, frames, amplitude, center, params["preplanned_speed"])
        pp_pos, pp_vel = agents.state_at(0)
        positions[pre_ids] = pp_pos
        velocities[pre_ids] = pp_vel

    # a preplanned agent is welded to its curve, so it can never also act as a
    # wander leader; draw leaders only from the boids left over
    leader_mask = np.zeros(n, dtype=bool)
    free = np.where(~mask)[0]
    n_leaders = min(cfg["n_leaders"], free.size)
    if n_leaders > 0:
        leader_mask[rng.choice(free, size=n_leaders, replace=False)] = True

    return World(positions, velocities, params, world_size,
                 obstacles=obstacles, predator_state=predator_state,
                 preplanned=agents, preplanned_mask=mask,
                 leader_mask=leader_mask, seed=run_seed)


def manifest_row(cfg, run_id, run_seed, frames, dt, world_size):
    return {
        "run_id": run_id,
        "seed": run_seed,
        "n_boids": cfg["n_boids"],
        "frames": frames,
        "dt": dt,
        "world_size": world_size,
        "w_separation": cfg["weights"]["separation"],
        "w_alignment": cfg["weights"]["alignment"],
        "w_cohesion": cfg["weights"]["cohesion"],
        "separation_radius": cfg["separation_radius"],
        "alignment_radius": cfg["alignment_radius"],
        "cohesion_radius": cfg["cohesion_radius"],
        "max_speed": cfg["max_speed"],
        "predator": int(cfg["predator"]),
        "obstacles": int(cfg["obstacles"]),
        "n_preplanned": cfg["n_preplanned"],
        "n_leaders": cfg["n_leaders"],
        "leader_follow_radius": cfg["leader_follow_radius"],
        "w_follow": cfg["w_follow"],
        "w_wander": cfg["w_wander"],
    }


def run_one(cfg, run_id, run_seed, frames, dt, world_size, out_dir, base=None):
    world = build_world(cfg, run_seed, frames, world_size, base)
    recorder = TrajectoryRecorder(dt)
    positions_seq = []
    velocities_seq = []
    leader_ids_seq = []
    lmask = world.leader_mask
    has_leaders = lmask is not None and lmask.any()
    for _ in range(frames):
        recorder.record(world.step_index, world)
        positions_seq.append(world.positions.copy())
        velocities_seq.append(world.velocities.copy())
        if has_leaders:
            _, lids = leader_follow(world.positions, lmask,
                                    world.params["leader_follow_radius"])
        else:
            lids = np.full(cfg["n_boids"], -1, dtype=int)
        leader_ids_seq.append(lids)
        world.step(dt)

    tag = "run_%04d" % run_id
    recorder.to_csv(os.path.join(out_dir, tag + ".csv"))

    mask = world.preplanned_mask
    if mask is None:
        mask = np.zeros(cfg["n_boids"], dtype=bool)
    np.savez_compressed(
        os.path.join(out_dir, tag + ".npz"),
        positions=np.stack(positions_seq),
        velocities=np.stack(velocities_seq),
        preplanned_mask=np.asarray(mask, dtype=bool),
        leader_mask=np.asarray(lmask, dtype=bool),
        leader_ids=np.stack(leader_ids_seq),
        n_boids=cfg["n_boids"],
        dt=dt,
        world_size=world_size,
        w_separation=cfg["weights"]["separation"],
        w_alignment=cfg["weights"]["alignment"],
        w_cohesion=cfg["weights"]["cohesion"],
        separation_radius=cfg["separation_radius"],
        alignment_radius=cfg["alignment_radius"],
        cohesion_radius=cfg["cohesion_radius"],
        max_speed=cfg["max_speed"],
        predator=int(cfg["predator"]),
        obstacles=int(cfg["obstacles"]),
        n_preplanned=cfg["n_preplanned"],
        n_leaders=cfg["n_leaders"],
        leader_follow_radius=cfg["leader_follow_radius"],
        w_follow=cfg["w_follow"],
        w_wander=cfg["w_wander"],
        seed=run_seed,
        frames=frames,
    )
    return manifest_row(cfg, run_id, run_seed, frames, dt, world_size)


def write_manifest(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def collect(runs, frames, dt, seed, out_dir, world_size=60.0, base=None):
    os.makedirs(out_dir, exist_ok=True)
    master = np.random.default_rng(seed)
    rows = []
    for run_id in range(runs):
        cfg = sample_params(master)
        run_seed = int(master.integers(0, 2 ** 31 - 1))
        rows.append(run_one(cfg, run_id, run_seed, frames, dt, world_size,
                            out_dir, base))
    write_manifest(rows, os.path.join(out_dir, "manifest.csv"))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Batch run the boids simulation over swept parameters "
                    "and save each run as CSV plus a compressed npz.")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--world-size", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default=None,
                        help="path to a config file (default: sim/config.yaml)")
    args = parser.parse_args()

    cfg = config_mod.load(args.config)
    # the sweep owns the numeric ranges, but which rules are switched on comes
    # from the config, so a rule disabled there stays disabled across every run
    base = config_mod.to_params(cfg)
    frames = cfg["motion"]["frames"] if args.frames is None else args.frames
    dt = cfg["motion"]["dt"] if args.dt is None else args.dt
    world_size = (cfg["world"]["size"] if args.world_size is None
                  else args.world_size)

    out_dir = args.out
    if out_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.normpath(
            os.path.join(here, "..", "..", "deliverables", "dataset"))

    rows = collect(args.runs, frames, dt, args.seed,
                   out_dir, world_size, base)
    print("Wrote %d runs to %s" % (len(rows), out_dir))


if __name__ == "__main__":
    main()
