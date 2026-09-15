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


def split_by_category(rows, column, held_out_values):
    # split_by_column calls float(), so it cannot split on bounds. exact
    # string match, and an unmatched value yields an empty test split rather
    # than an error, so a typo shows up in the reported sizes
    held = set(held_out_values)
    train, test = [], []
    for row in rows:
        target = test if row[column] in held else train
        target.append(_run_id(row))
    return sorted(train), sorted(test)
