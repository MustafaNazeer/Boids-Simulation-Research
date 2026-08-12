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


def aggregate(results):
    """Combine per seed comparison dicts into per arm, per relation spreads."""
    if not results:
        raise ValueError("no results to aggregate")

    out = {}
    for arm in ARMS:
        out[arm] = {}
        for relation in labels.RELATIONS:
            out[arm][relation] = {
                "auc": _spread([r[arm][relation]["auc"] for r in results]),
                "f1": _spread([r[arm][relation]["f1"] for r in results]),
            }

    out["recall"] = {
        relation: _spread([r["recall"][relation] for r in results])
        for relation in labels.RELATIONS
    }
    return out


def load_and_aggregate(paths_to_json):
    results = []
    for path in paths_to_json:
        with open(path) as f:
            results.append(json.load(f))
    return aggregate(results)


def main():
    parser = argparse.ArgumentParser(
        description="Run one seed of the comparison and dump it as JSON.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir or paths.fixture_dir()
    result = report.run_comparison(run_dir, epochs=args.epochs,
                                   stride=args.stride, seed=args.seed)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
