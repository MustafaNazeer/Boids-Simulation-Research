import os
import csv
import argparse
import numpy as np

from sim.boids import collect
from sim.boids import config as config_mod

# the config module already owns this tuple, so it is not restated here
BOUNDS_MODES = config_mod.BOUNDS_MODES


def bounds_stream(seed):
    # bounds must not come from the master rng. inserting a third draw into
    # the per run loop would shift every later parameter, and the planner
    # would stop reproducing the frozen collect's physics values. a spawned
    # child is an independent stream, and the child at a given spawn index is
    # the same however many children are requested
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])


def plan_rows(runs, frames, dt, world_size, seed):
    # the master rng is built and drawn from exactly as collect.collect does
    master = np.random.default_rng(seed)
    brng = bounds_stream(seed)
    rows = []
    for run_id in range(runs):
        cfg = collect.sample_params(master)
        run_seed = int(master.integers(0, 2 ** 31 - 1))
        bounds = BOUNDS_MODES[int(brng.integers(0, len(BOUNDS_MODES)))]
        rows.append(collect.manifest_row(cfg, run_id, run_seed, frames, dt,
                                         world_size, bounds))
    return rows


def row_to_cfg(row):
    # the inverse of collect.manifest_row for the fields run_one consumes.
    # execution reads rows from the manifest file, so a task can run rows it
    # did not plan
    return {
        "n_boids": int(row["n_boids"]),
        "weights": {
            "separation": float(row["w_separation"]),
            "alignment": float(row["w_alignment"]),
            "cohesion": float(row["w_cohesion"]),
        },
        "separation_radius": float(row["separation_radius"]),
        "alignment_radius": float(row["alignment_radius"]),
        "cohesion_radius": float(row["cohesion_radius"]),
        "max_speed": float(row["max_speed"]),
        "predator": bool(int(row["predator"])),
        "obstacles": bool(int(row["obstacles"])),
        "n_preplanned": int(row["n_preplanned"]),
        "n_leaders": int(row["n_leaders"]),
        "leader_follow_radius": float(row["leader_follow_radius"]),
        "w_follow": float(row["w_follow"]),
        "w_wander": float(row["w_wander"]),
        "n_food": int(row["n_food"]),
        "food_sensing_radius": float(row["food_sensing_radius"]),
        "w_food_seek": float(row["w_food_seek"]),
    }


def read_manifest(out_dir):
    path = os.path.join(out_dir, "manifest.csv")
    if not os.path.exists(path):
        raise SystemExit(
            "no manifest at %s. Run `sweep plan` before `sweep run`." % path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def shard_indices(n_runs, shard_index, shard_count):
    if shard_count < 1:
        raise ValueError("shard-count must be at least 1, got %r"
                         % (shard_count,))
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            "shard-index must be in [0, %d), got %r"
            % (shard_count, shard_index))
    # strided rather than contiguous, so the runs whose csv is retained are
    # spread across tasks instead of landing entirely on task 0
    return list(range(shard_index, n_runs, shard_count))


def execute(rows, out_dir, indices, base=None, csv_runs=0):
    os.makedirs(out_dir, exist_ok=True)
    done = []
    for i in indices:
        row = rows[i]
        run_id = int(row["run_id"])
        collect.run_one(
            row_to_cfg(row), run_id, int(row["seed"]),
            int(row["frames"]), float(row["dt"]), float(row["world_size"]),
            out_dir, base, bounds=row["bounds"],
            # the retained csv follows the run id, not the shard, so the same
            # runs keep their csv however the sweep is divided up
            write_csv=run_id < csv_runs)
        done.append(run_id)
    return done


def missing(rows, out_dir):
    absent = []
    for row in rows:
        tag = "run_%04d" % int(row["run_id"])
        if not os.path.exists(os.path.join(out_dir, tag + ".npz")):
            absent.append(int(row["run_id"]))
    return absent


def _resolve_settings(args):
    cfg = config_mod.load(args.config)
    base = config_mod.to_params(cfg)
    frames = cfg["motion"]["frames"] if args.frames is None else args.frames
    dt = cfg["motion"]["dt"] if args.dt is None else args.dt
    world_size = (cfg["world"]["size"] if args.world_size is None
                  else args.world_size)
    # world.bounds is deliberately ignored. the sweep samples it per run
    return base, frames, dt, world_size


def _add_config_args(p):
    p.add_argument("--config", default=None,
                   help="path to a config file (default: sim/config.yaml)")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--world-size", type=float, default=None)


def main():
    parser = argparse.ArgumentParser(
        description="Plan, execute, and verify a shardable parameter sweep "
                    "of the boids simulation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser(
        "plan", help="draw the whole sweep and write manifest.csv")
    p_plan.add_argument("--runs", type=int, required=True)
    p_plan.add_argument("--seed", type=int, default=20260820)
    p_plan.add_argument("--out", required=True)
    _add_config_args(p_plan)

    p_run = sub.add_parser("run", help="execute one shard of a planned sweep")
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--shard-index", type=int, required=True)
    p_run.add_argument("--shard-count", type=int, required=True)
    p_run.add_argument("--csv-runs", type=int, default=0,
                       help="retain the csv for run ids below this number")
    _add_config_args(p_run)

    p_verify = sub.add_parser(
        "verify", help="report planned runs that have no npz")
    p_verify.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.command == "plan":
        base, frames, dt, world_size = _resolve_settings(args)
        os.makedirs(args.out, exist_ok=True)
        rows = plan_rows(args.runs, frames, dt, world_size, args.seed)
        collect.write_manifest(rows, os.path.join(args.out, "manifest.csv"))
        shares = {m: sum(1 for r in rows if r["bounds"] == m)
                  for m in BOUNDS_MODES}
        print("Planned %d runs of %d frames into %s"
              % (len(rows), frames, args.out))
        print("Bounds: %s" % shares)
        return

    if args.command == "run":
        base, _, _, _ = _resolve_settings(args)
        rows = read_manifest(args.out)
        indices = shard_indices(len(rows), args.shard_index, args.shard_count)
        done = execute(rows, args.out, indices, base, csv_runs=args.csv_runs)
        print("Shard %d of %d completed %d runs: %s"
              % (args.shard_index, args.shard_count, len(done), done))
        return

    rows = read_manifest(args.out)
    absent = missing(rows, args.out)
    if absent:
        raise SystemExit("Missing %d of %d runs: %s"
                         % (len(absent), len(rows), absent))
    print("All %d planned runs are present in %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
