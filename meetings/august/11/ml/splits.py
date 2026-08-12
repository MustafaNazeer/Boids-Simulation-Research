import csv
import os
import random


def read_manifest(run_dir):
    with open(os.path.join(run_dir, "manifest.csv")) as f:
        return list(csv.DictReader(f))


def _run_id(row):
    return "run_%04d" % int(row["run_id"])


def split_by_column(rows, column, threshold):
    train, test = [], []
    for row in rows:
        target = train if float(row[column]) < float(threshold) else test
        target.append(_run_id(row))
    return train, test


def random_split(rows, fraction, seed):
    ids = sorted(_run_id(r) for r in rows)
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(round(len(ids) * fraction))
    return sorted(ids[:cut]), sorted(ids[cut:])
