"""Repeat the comparison across seeds and aggregate.

RESULTS.md from the single seed run flagged that one seed, one split and one
dataset carry no error bars. This module supplies them.

The seed drives both the train and test split and the model initialisation, so
the spread reported here is the total experiment to experiment variance, which
is the honest answer to "if we ran this again, what would we get".

Each seed is run as its own process so several can go at once. `aggregate`
then combines the dumped JSON files.
"""

import argparse
import json

import numpy as np

from ml import paths, report
from ml.data import labels

ARMS = ["model", "ablation_leaky", "ablation_no_position", "distance",
        "oracle"]


def _spread(values):
    values = [float(v) for v in values]
    return {
        "mean": float(np.mean(values)),
        # sample standard deviation, so a single seed reports 0.0 rather than
        # raising, and two or more seeds get the unbiased estimate
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }


def _aggregate_block(results, pick):
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for relation in labels.RELATIONS:
            out[arm][relation] = {
                "auc": _spread([pick(r)[arm][relation]["auc"] for r in results]),
                "f1": _spread([pick(r)[arm][relation]["f1"] for r in results]),
            }

    out["recall"] = {
        relation: _spread([pick(r)["recall"][relation] for r in results])
        for relation in labels.RELATIONS
    }
    return out


def aggregate(results):
    """Combine per seed comparison dicts into per arm, per relation spreads.

    A "transfer" block is summarised the same way when every seed carries one,
    which is what the held out bounds experiment produces. It is omitted
    entirely if any seed lacks it, so a mixed set of runs cannot silently
    average a partial result.
    """
    if not results:
        raise ValueError("no results to aggregate")

    out = _aggregate_block(results, lambda r: r)
    if all("transfer" in r for r in results):
        out["transfer"] = _aggregate_block(results, lambda r: r["transfer"])
    return out


def load_and_aggregate(paths_to_json):
    results = []
    for path in paths_to_json:
        with open(path) as f:
            results.append(json.load(f))
    return aggregate(results)


def aggregate_architectures(results):
    """Combine per seed architecture comparisons into per arm spreads.

    Refuses a mixed set of runs rather than averaging over whatever each seed
    happens to contain. A table whose rows silently carry different seed counts
    is worse than no table, because nothing on its face says so.
    """
    if not results:
        raise ValueError("no results to aggregate")

    names = set(results[0])
    for r in results[1:]:
        if set(r) != names:
            raise ValueError(
                "every seed must carry the same arms; got %s against %s"
                % (sorted(set(r)), sorted(names)))

    out = {}
    for name in sorted(names):
        out[name] = {
            "auc": {
                relation: _spread([r[name]["scores"][relation]["auc"]
                                   for r in results])
                for relation in labels.RELATIONS
            },
            "f1": {
                relation: _spread([r[name]["scores"][relation]["f1"]
                                   for r in results])
                for relation in labels.RELATIONS
            },
            # a property of the arm rather than of the seed, so carried
            # through rather than averaged
            "params": results[0][name]["params"],
            "seconds": _spread([r[name]["seconds"] for r in results]),
        }
    return out


def load_and_aggregate_architectures(paths_to_json):
    results = []
    for path in paths_to_json:
        with open(path) as f:
            results.append(json.load(f))
    return aggregate_architectures(results)


def main():
    parser = argparse.ArgumentParser(
        description="Run one seed of the comparison and dump it as JSON.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--out", required=True)
    parser.add_argument("--architectures", action="store_true",
                        help="run the architecture comparison instead of the "
                             "standard one: one model per arm from ml.arms, "
                             "every arm position free.")
    parser.add_argument("--arms", default=None,
                        help="comma separated arm names to run, defaulting to "
                             "all thirteen. Use this to run one axis at a "
                             "time.")
    parser.add_argument("--agent-types", action="store_true",
                        help="run the agent type comparison instead of the "
                             "standard one: plain, typed, and noise arms, "
                             "each scored overall and on special pairs only.")
    parser.add_argument("--held-out-bounds", default=None,
                        help="comma separated bounds modes to reserve for the "
                             "transfer evaluation, for example 'none'. The "
                             "model never trains on them and is scored on "
                             "them without retraining.")
    args = parser.parse_args()

    run_dir = args.run_dir or paths.fixture_dir()
    if args.architectures:
        names = ([n.strip() for n in args.arms.split(",") if n.strip()]
                 if args.arms else None)
        result = report.run_architecture_comparison(
            run_dir, arm_names=names, epochs=args.epochs,
            stride=args.stride, seed=args.seed)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print("wrote %s" % args.out)
        return

    if args.agent_types:
        result = report.run_agent_type_comparison(
            run_dir, epochs=args.epochs, stride=args.stride, seed=args.seed)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print("wrote %s" % args.out)
        return

    kwargs = {}
    if args.held_out_bounds:
        modes = [m.strip() for m in args.held_out_bounds.split(",") if m.strip()]
        rows = report.splits.read_manifest(run_dir)
        kwargs["split"] = report.split_within(modes, fraction=0.8)
        kwargs["extra_test_ids"] = report.ids_with_bounds(rows, modes)
        if not kwargs["extra_test_ids"]:
            raise SystemExit("no runs with bounds in %s, nothing to transfer to"
                             % modes)
    result = report.run_comparison(run_dir, epochs=args.epochs,
                                   stride=args.stride, seed=args.seed,
                                   **kwargs)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
