"""The laboratory midge swarms of Sinhuber et al., Scientific Data 2019.

Columns, as probed on Ob11.txt: track id, x, y, z, time, vx, vy, vz, ax, ay,
az, comma separated, sampled every 0.01 s. Per the dataset paper, positions
are in millimetres and time in seconds.
"""

import json
import os
import re
import urllib.request

import numpy as np

from discover import data

ARTICLE = "https://api.figshare.com/v2/articles/7336193/versions/2"
DT = 0.01
TEST_SWARMS = ("Ob4", "Ob9", "Ob14", "Ob19")
# Frames per midge window, set by Mustafa on 2026-10-01 after the free window
# length investigation (see RESULTS-discover.md, Frozen choices). The
# simulated systems keep 21 frames.
MIDGE_WINDOW = 81
# Only the swarm track files are read; anything else in the raw directory
# (a README, a partial download) is ignored.
SWARM_FILE = re.compile(r"Ob\d+\.txt")


def parse(path):
    a = np.loadtxt(path, delimiter=",")
    frames = np.rint(a[:, 4] / DT).astype(int)
    frames -= frames.min()
    tracks, col = np.unique(a[:, 0].astype(int), return_inverse=True)
    P = np.full((frames.max() + 1, len(tracks), 3), np.nan)
    V = np.full_like(P, np.nan)
    P[frames, col] = a[:, 1:4]
    V[frames, col] = a[:, 5:8]
    return P, V


def fetch(dest_dir):
    with urllib.request.urlopen(ARTICLE) as r:
        listing = json.load(r)
    paths = []
    for f in listing["files"]:
        path = os.path.join(dest_dir, f["name"])
        if not (os.path.exists(path) and os.path.getsize(path) == f["size"]):
            urllib.request.urlretrieve(f["download_url"], path)
        if os.path.getsize(path) != f["size"]:
            raise IOError("%s is %d bytes, listing says %d"
                          % (path, os.path.getsize(path), f["size"]))
        paths.append(path)
    return paths


def prepare(raw_dir, out_dir, length=MIDGE_WINDOW, per_split=20, seed=20261001):
    """Write the fit, val and test windows.

    Within each training swarm, consecutive windows in time order alternate
    fit, val, fit, val, as the spec says, and whole (fit, val) pairs are
    sampled, so the two sets come from the same stretches of the same swarms.
    Pairs are taken in the order their windows first appear in one seeded
    permutation of the training windows. That order is a uniform random
    order of the pairs, and it draws from the generator exactly as the
    earlier window sampling did, so the test windows stay the same.
    """
    train, pairs, test = [], [], []
    names = [n for n in os.listdir(raw_dir) if SWARM_FILE.fullmatch(n)]
    for name in sorted(names):
        swarm = name.split(".")[0]
        P, V = parse(os.path.join(raw_dir, name))
        ws = data.cut_windows(P, V, DT, swarm, length)
        if swarm in TEST_SWARMS:
            test.extend(ws)
            continue
        for k in range(0, len(ws) - 1, 2):
            pairs.append((len(train) + k, len(train) + k + 1))
        train.extend(ws)
    pair_of = {}
    for p, (a, b) in enumerate(pairs):
        pair_of[a] = pair_of[b] = p
    rng = np.random.default_rng(seed)
    chosen = []
    for i in rng.permutation(len(train)):
        p = pair_of.get(int(i))
        if p is not None and p not in chosen:
            chosen.append(p)
    if len(chosen) < per_split:
        print("warning: only %d (fit, val) pairs available, fewer than the "
              "%d asked for" % (len(chosen), per_split))
    chosen = chosen[:per_split]
    fit = [train[pairs[p][0]] for p in chosen]
    val = [train[pairs[p][1]] for p in chosen]
    order = rng.permutation(len(test))
    held = [test[i] for i in order[:2 * per_split]]
    data.save_windows(fit, os.path.join(out_dir, "midges_fit.npz"))
    data.save_windows(val, os.path.join(out_dir, "midges_val.npz"))
    data.save_windows(held, os.path.join(out_dir, "midges_test.npz"))
