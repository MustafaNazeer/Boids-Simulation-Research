"""Entry points. Every paid command checks the gates and the key first."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from discover import (baseline, data, ledger, llm, prompts, sandbox, search,
                      truth)

CAP = 20.0
TASK_CAPS = {"sim-boids": 3.0, "sim-well": 3.0, "midges": 10.0,
             "boids-haiku": 1.0, "boids-sonnet": 2.0, "boids-opus": 4.0}
TIMEOUTS = {"sim-boids": 10.0, "sim-well": 10.0, "midges": 60.0,
            "boids-haiku": 10.0, "boids-sonnet": 10.0, "boids-opus": 10.0}
TASK_MODELS = {"sim-boids": "claude-sonnet-5-5",
               "sim-well": "claude-sonnet-5-5",
               "midges": "claude-sonnet-5-5",
               "boids-haiku": "claude-haiku-4-5",
               "boids-sonnet": "claude-sonnet-5-5",
               "boids-opus": "claude-opus-5-5"}
# The model comparison tasks search the sim-boids data, noise 0.
DATA_TASK = {"boids-haiku": "sim-boids", "boids-sonnet": "sim-boids",
             "boids-opus": "sim-boids"}
HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = {"boids": (truth.BOIDS, truth.BOIDS_PARAMS, truth.BOIDS_WEIGHTS, 0.0),
           "well": (truth.WELL, truth.WELL_PARAMS, truth.WELL_WEIGHTS, 2.0)}


def gates_pass():
    tests = [os.path.join(HERE, "tests", n) for n in
             ("test_gates.py", "test_sandbox.py", "test_ledger.py",
              "test_search.py::test_stops_when_the_ledger_refuses")]
    root = os.path.dirname(HERE)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", *tests],
                          cwd=root).returncode == 0


def has_credentials():
    return shutil.which("ant") is not None and subprocess.run(
        ["ant", "auth", "status"], capture_output=True).returncode == 0


def _paths(root, task, noise=0.0):
    task = DATA_TASK.get(task, task)
    stem = "midges" if task == "midges" else task.replace("-", "_")
    if noise:
        stem += "_n%g" % noise
    return (os.path.join(root, "data", stem + "_fit.npz"),
            os.path.join(root, "data", stem + "_val.npz"))


def prepare_sim(root, system, noise):
    source, params, weights, kick = SYSTEMS[system]
    P, V = truth.simulate(source, params, weights, n_agents=20, frames=841,
                          dt=0.05, seed=20261001, kick=kick)
    if not np.isfinite(P).all() or np.abs(P).max() > 1e3:
        sys.exit("simulation of %s left bounds; report before running" % system)
    if noise:
        rng = np.random.default_rng(20261002)
        P = P + rng.normal(scale=noise * P.std(), size=P.shape)
        V = np.vstack([V[:1], np.diff(P, axis=0) / 0.05])
    ws = data.cut_windows(P, V, 0.05, "sim-" + system, length=21)
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    fit, val = _paths(root, "sim-" + system, noise)
    data.save_windows(ws[0::2][:20], fit)
    data.save_windows(ws[1::2][:20], val)
    print("wrote", fit, val, "noise", noise)


def scale_text(fit_path):
    """The prompt's scale facts, with the radius floor the child applies."""
    windows = data.load_windows(fit_path)
    return prompts.scale_facts(windows, baseline.radius_range(windows)[0])


def uncommitted(path):
    """True if git reports path as modified, staged or untracked.

    The check is skipped (False) when git is missing or path is not inside a
    work tree, so test-once still runs on a machine without the repo.
    """
    try:
        proc = subprocess.run(["git", "status", "--porcelain", "--", path],
                              cwd=os.path.dirname(path), capture_output=True,
                              text=True)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=HERE)
    ap = argparse.ArgumentParser(prog="discover")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-sim", parents=[common])
    p.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    p.add_argument("--noise", type=float, default=0.0)
    p.set_defaults(task=None)
    sub.add_parser("gates", parents=[common]).set_defaults(task=None, noise=0.0)
    p = sub.add_parser("search", parents=[common])
    p.add_argument("--task", choices=sorted(TASK_CAPS), required=True)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--generations", type=int, default=1)
    p = sub.add_parser("baseline", parents=[common])
    p.add_argument("--task", choices=sorted(TASK_CAPS), required=True)
    p.add_argument("--noise", type=float, default=0.0)
    sub.add_parser("test-once", parents=[common]).set_defaults(task=None,
                                                               noise=0.0)
    return ap


def main(argv=None):
    a = _parser().parse_args(argv)
    # The ledger and the test marker guard money and the one test run, so
    # they live beside the package whatever --root says. --root only moves
    # the data and the search logs.
    pinned = os.path.join(HERE, "runs")
    os.makedirs(pinned, exist_ok=True)
    runs = os.path.join(a.root, "runs")
    os.makedirs(runs, exist_ok=True)

    if a.noise and a.task == "midges":
        sys.exit("--noise applies only to the simulated tasks")
    if a.noise and a.task in DATA_TASK:
        sys.exit("--noise is refused for the model comparison tasks")

    if a.command == "prepare-sim":
        prepare_sim(a.root, a.system, a.noise)
    elif a.command == "gates":
        sys.exit(0 if gates_pass() else "gates failed")
    elif a.command == "search":
        if not gates_pass():
            sys.exit("gates failed: nothing paid runs until they pass")
        if not (os.environ.get("ANTHROPIC_API_KEY") or has_credentials()):
            sys.exit("no API key: export ANTHROPIC_API_KEY or run ant auth login")
        led = ledger.Ledger(os.path.join(pinned, "ledger.jsonl"), CAP)
        log = os.path.join(runs, "%s-%d.jsonl" % (a.task, int(time.time())))
        fit, val = _paths(a.root, a.task, a.noise)
        scale = scale_text(fit)
        client = llm.BatchClient(model=TASK_MODELS[a.task])
        found = search.run(a.task, TASK_CAPS[a.task], fit, val,
                           client, led, log, a.generations,
                           scale_text=scale, timeout=TIMEOUTS[a.task])
        ok = sorted((c for c in found if c.ok), key=lambda c: -c.score)
        print("log", log, "candidates", len(found), "ok", len(ok),
              "spent so far %.4f USD" % led.committed())
        if ok:
            print("best", ok[0].score, ok[0].val_r2, ok[0].params, ok[0].weights)
    elif a.command == "baseline":
        fit, val = _paths(a.root, a.task, a.noise)
        out = baseline.fit(data.load_windows(fit), data.load_windows(val))
        print(json.dumps(out))
    elif a.command == "test-once":
        marker = os.path.join(pinned, "test_done.json")
        if os.path.exists(marker):
            sys.exit("the test set was already used; see " + marker)
        best_path = os.path.join(pinned, "best_midges.py")
        if uncommitted(best_path):
            sys.exit("%s has uncommitted changes; commit it so the tested "
                     "program is on record" % best_path)
        fit, val = _paths(a.root, "midges")
        test = os.path.join(a.root, "data", "midges_test.npz")
        with open(best_path) as f:
            source = f.read()
        # Claim the marker before the test windows are read. O_EXCL makes
        # the claim atomic, and a run that crashes after claiming still
        # counts as the one use of the test set.
        try:
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            sys.exit("the test set was already used; see " + marker)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"status": "started", "time": time.time()}))
            f.flush()
            # Nothing is chosen on the test windows: the candidate's
            # parameters and weights come from fit, and the baseline's radii,
            # threshold and kept terms come from fit and val before test is
            # scored once.
            llm_out = sandbox.evaluate(source, fit, test, timeout=60.0)
            base_out = baseline.fit_frozen(data.load_windows(fit),
                                           data.load_windows(val),
                                           data.load_windows(test))
            result = {"llm": llm_out, "baseline": base_out,
                      "time": time.time()}
            f.seek(0)
            f.truncate()
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
