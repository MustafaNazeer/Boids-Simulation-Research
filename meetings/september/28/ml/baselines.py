"""Distance only references.

The labels are distance thresholds, so a model handed distance can score well
while learning nothing about interaction. These two references make that
visible.

`distance_reference` scores every pair by negated distance. For a single
monotone feature this is equivalent to fitting a logistic regression on
distance: the AUC is identical, because AUC depends only on the ranking.

`oracle_threshold_reference` picks the best threshold separately per run,
using that run's own labels. It is cheating by construction. Its purpose is to
be the ceiling on what any purely distance based method can reach. Always
label it as an oracle when reporting it.
"""

import numpy as np

from ml import metrics
from ml.data import labels


def _collect(dataset):
    per_run = {}
    for sample in dataset:
        distances = sample.edge_attr[:, 0].numpy()
        targets = sample.y.numpy()
        bucket = per_run.setdefault(sample.run_id, {"d": [], "y": []})
        bucket["d"].append(distances)
        bucket["y"].append(targets)
    return {run: {"d": np.concatenate(v["d"]),
                  "y": np.concatenate(v["y"], axis=0)}
            for run, v in per_run.items()}


def distance_reference(dataset):
    per_run = _collect(dataset)
    all_d = np.concatenate([v["d"] for v in per_run.values()])
    all_y = np.concatenate([v["y"] for v in per_run.values()], axis=0)

    report = {}
    for col, relation in enumerate(labels.RELATIONS):
        scores = -all_d
        f1, _ = metrics.best_f1(scores, all_y[:, col])
        report[relation] = {
            "auc": metrics.roc_auc(scores, all_y[:, col]),
            "f1": f1,
        }
    return report


def oracle_threshold_reference(dataset):
    per_run = _collect(dataset)

    report = {}
    for col, relation in enumerate(labels.RELATIONS):
        aucs, f1s, weights = [], [], []
        for value in per_run.values():
            scores = -value["d"]
            targets = value["y"][:, col]
            auc = metrics.roc_auc(scores, targets)
            f1, _ = metrics.best_f1(scores, targets)
            if np.isfinite(auc):
                aucs.append(auc)
            f1s.append(f1)
            weights.append(len(targets))
        report[relation] = {
            "auc": float(np.mean(aucs)) if aucs else float("nan"),
            "f1": float(np.average(f1s, weights=weights)),
        }
    return report
