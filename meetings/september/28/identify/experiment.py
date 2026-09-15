"""The sampling rate experiment: how the recovery threshold moves with dt.

Design and registered predictions:
`docs/superpowers/specs/2026-09-14-rule-identifiability-design.md`.

The claim under test is that, with positions observed under noise sigma and
velocities obtained by differencing them, the noise level at which the weights
stop being recoverable scales as

    sigma* = |a| * dt^2 / sqrt(6)

so at a fixed frame count the exponent of sigma* in dt is 2, and at a fixed
observation duration, where halving dt doubles the frame count, it is 1.5.

Nothing is emitted unless all three tripwires pass. That is the whole point of
having them: without the tripwires, "the exponent is zero" and "the estimator
is broken" are the same observation.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from identify import estimator, observe, tripwires

SQRT6 = np.sqrt(6.0)

# the crossing used to define sigma*, as a relative error on the weights
THRESHOLD = 0.20


def _f(run, key):
    return float(run[key])


def _truth(run):
    return (np.array([_f(run, "separation_radius"),
                      _f(run, "alignment_radius"),
                      _f(run, "cohesion_radius")]),
            np.array([_f(run, "w_separation"),
                      _f(run, "w_alignment"),
                      _f(run, "w_cohesion")]))


def signal_scale(run, t0, n_frames, stride=1):
    """RMS per component of the true velocity increment, the |a| * dt above.

    Measured per run rather than assumed, because changing dt changes the
    trajectory and not only its sampling, so the arms are not the same
    physical system and a shared constant would hide that.

    It must follow the same anchors the fit used, stride included. A thinned
    window covers a wider span and therefore a different stretch of the
    trajectory, and dividing by a signal measured somewhere else would put
    that difference straight into the collapsed group.
    """
    V = run["velocities"]
    idx = np.arange(t0, t0 + (n_frames - 1) * stride + 1, stride)
    inc = V[idx + 1] - V[idx]
    return float(np.sqrt(np.mean(inc * inc)))


def _true_velocity_window(run, sigma, seed, t0, n_frames, stride=1):
    """The control arm: noisy positions, exact velocities.

    The claim attributes the whole dt dependence to differentiating noisy
    positions, which multiplies the noise by sqrt(6) / dt. Handing the
    estimator the recorded velocities removes that term and nothing else, so
    if the exponent survives here the mechanism is misattributed even if the
    effect is real. This is the control most able to embarrass the claim.
    """
    from identify import basis
    P = observe.add_position_noise(run["positions"], sigma,
                                   np.random.default_rng(seed))
    V = run["velocities"]
    frames = []
    targets = []
    for t in range(t0, t0 + (n_frames - 1) * stride + 1, stride):
        diff, dist = basis.pairwise(P[t])
        frames.append(observe.Frame(P[t], V[t], diff, dist))
        targets.append(V[t + 1] - V[t])
    return frames, np.concatenate(targets)


def one_fit(run, sigma, seed, t0, n_frames, with_reference=False,
            true_velocities=False, stride=1):
    P = run["positions"]
    dt = _f(run, "dt")
    radii_true, weights_true = _truth(run)
    if true_velocities:
        frames, target = _true_velocity_window(run, sigma, seed, t0, n_frames,
                                               stride)
    else:
        rng = np.random.default_rng(seed)
        obs = observe.observed(P, dt, sigma, rng)
        frames, target = obs.window(t0=t0, n_frames=n_frames, stride=stride)
    keep = estimator.unclamped_mask(run["velocities"], _f(run, "max_speed"),
                                    0.0, t0=t0 + 1, n_frames=n_frames,
                                    stride=stride)
    fit = estimator.estimate(frames, target, keep, dt)

    radius_err = estimator.relative_error(fit.radii, radii_true)
    weight_err = estimator.relative_error(fit.weights, weights_true)
    row = {
        "run_id": int(run["seed"]),
        "dt": dt,
        "n_boids": int(run["n_boids"]),
        "frames": int(run["frames"]),
        "t0": t0,
        "n_frames": n_frames,
        "stride": stride,
        "span_frames": (n_frames - 1) * stride + 1,
        "sigma": sigma,
        "noise_seed": seed,
        "true_velocities": bool(true_velocities),
        "radii_true": radii_true.tolist(),
        "radii_hat": fit.radii.tolist(),
        "weights_true": weights_true.tolist(),
        "weights_hat": fit.weights.tolist(),
        "radius_rel_err": radius_err.tolist(),
        "weight_rel_err": weight_err.tolist(),
        "radius_rel_err_median": float(np.median(radius_err)),
        "weight_rel_err_median": float(np.median(weight_err)),
        "residual_rmse": fit.rmse,
        "signal_scale": signal_scale(run, t0, n_frames, stride),
        "clamped_fraction": float(1.0 - keep.mean()),
        "interval_widths": [
            float(hi - lo) for lo, hi in
            (estimator.identified_interval(frames, r) for r in radii_true)],
    }
    if with_reference:
        row["separation_only_radius"] = estimator.separation_only_radius(
            frames, target, keep, dt)
    return row


def crossing(sigmas, errors, threshold=THRESHOLD):
    """sigma* by log linear interpolation on the first upward crossing.

    Returns None when the curve never crosses inside the swept range, which
    is a real outcome and must be reported as such rather than clipped to an
    endpoint.
    """
    order = np.argsort(sigmas)
    s = np.asarray(sigmas, dtype=float)[order]
    e = np.asarray(errors, dtype=float)[order]
    for i in range(1, len(s)):
        if e[i - 1] < threshold <= e[i]:
            if e[i] == e[i - 1]:
                return float(s[i])
            frac = (np.log(threshold) - np.log(max(e[i - 1], 1e-12))) / \
                   (np.log(e[i]) - np.log(max(e[i - 1], 1e-12)))
            return float(np.exp(np.log(s[i - 1]) +
                                frac * (np.log(s[i]) - np.log(s[i - 1]))))
    return None


def exponent(dts, sigma_stars):
    """Slope of log sigma* against log dt, with its standard error."""
    pairs = [(d, s) for d, s in zip(dts, sigma_stars) if s is not None]
    if len(pairs) < 2:
        return None
    x = np.log([d for d, _ in pairs])
    y = np.log([s for _, s in pairs])
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    if n == 2:
        return {"slope": float(slope), "stderr": None, "n": n}
    resid = y - (slope * x + intercept)
    dof = n - 2
    s_err = np.sqrt(np.sum(resid ** 2) / dof / np.sum((x - x.mean()) ** 2))
    return {"slope": float(slope), "stderr": float(s_err), "n": n}


def load_runs(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "run_*.npz")))
    if not paths:
        raise SystemExit("no run_*.npz under %s" % run_dir)
    return [np.load(p) for p in paths]


def frames_for_run(run, n_frames, duration, t0):
    """How many frames this run contributes, under whichever protocol.

    `duration` selects protocol 2: every arm observes the same simulated time
    rather than the same number of frames, so a halved timestep doubles the
    frame count and buys a factor of sqrt(2) in averaging. That is the whole
    reason the two protocols predict different exponents.
    """
    if duration is not None:
        want = int(round(duration / _f(run, "dt")))
    elif n_frames is not None:
        want = n_frames
    else:
        want = int(run["frames"]) - t0 - 1
    return min(want, int(run["frames"]) - t0 - 1)


def sweep(run_dir, sigmas, noise_seeds, t0=1, n_frames=None, duration=None,
          out=None, skip_tripwires=False, true_velocities=False,
          shard_index=0, shard_count=1, label=None, stride=1,
          only_dt=None):
    runs = load_runs(run_dir)
    if not skip_tripwires:
        # every tripwire, on a real pair of runs from this very dataset
        tripwires.require(runs[0], runs[-1], n_frames=8)

    # strided sharding, the same scheme sweep.py uses, so each shard spans the
    # timestep groups rather than taking one of them whole
    if only_dt is not None:
        runs = [r for r in runs if abs(_f(r, "dt") - only_dt) < 1e-12]
        if not runs:
            raise SystemExit("no runs at dt=%g in %s" % (only_dt, run_dir))
    mine = list(range(shard_index, len(runs), shard_count))

    rows = []
    started = time.time()
    for index in mine:
        run = runs[index]
        frames_here = frames_for_run(run, n_frames, duration, t0)
        for sigma in sigmas:
            for k, seed in enumerate(noise_seeds):
                rows.append(one_fit(run, float(sigma), int(seed), t0,
                                    frames_here, with_reference=(k == 0),
                                    true_velocities=true_velocities,
                                    stride=stride))
    elapsed = time.time() - started

    payload = {
        "run_dir": run_dir,
        "label": label,
        "sigmas": list(map(float, sigmas)),
        "noise_seeds": list(map(int, noise_seeds)),
        "t0": t0,
        "n_frames": n_frames,
        "duration": duration,
        "stride": stride,
        "only_dt": only_dt,
        "true_velocities": bool(true_velocities),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "run_indices": mine,
        "wall_clock_seconds": elapsed,
        "threshold": THRESHOLD,
        "rows": rows,
    }
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def summarise(payload):
    """Per timestep sigma*, and the exponent of sigma* in dt."""
    by_dt = {}
    for row in payload["rows"]:
        by_dt.setdefault(row["dt"], {}).setdefault(row["sigma"], []).append(
            row["weight_rel_err_median"])
    out = {"per_dt": {}, "threshold": payload["threshold"]}
    dts = []
    stars = []
    for dt in sorted(by_dt):
        sigmas = sorted(by_dt[dt])
        errs = [float(np.median(by_dt[dt][s])) for s in sigmas]
        star = crossing(sigmas, errs, payload["threshold"])
        scales = [r["signal_scale"] for r in payload["rows"] if r["dt"] == dt]
        scale = float(np.median(scales))
        out["per_dt"][dt] = {
            "sigmas": sigmas,
            "median_weight_rel_err": errs,
            "sigma_star": star,
            "signal_scale": scale,
            # the collapsed group the spec names as the decisive row
            "collapsed": (None if star is None
                          else float(star * SQRT6 / (scale * dt))),
        }
        dts.append(dt)
        stars.append(star)
    out["exponent"] = exponent(dts, stars)
    return out


def signed_bias(rows, dt, sigma, tol=1e-12):
    """Mean signed relative error of each weight, over runs and noise seeds.

    Signed, not absolute, because the whole point is to separate a systematic
    displacement from scatter. The spread of these same numbers across noise
    seeds is what says which of the two dominates.
    """
    sub = [r for r in rows
           if abs(r["dt"] - dt) < tol and abs(r["sigma"] - sigma) < tol]
    if not sub:
        return None
    err = np.array([(np.array(r["weights_hat"]) - np.array(r["weights_true"]))
                    / np.array(r["weights_true"]) for r in sub])
    return err.mean(axis=0)


def _loglog_fit(x, y, degree=1):
    if len(x) < degree + 1:
        return None
    return np.polyfit(np.log(x), np.log(y), degree)


def bias_surface(rows, band=(0.01, 0.5)):
    """Fit the bias as a power law in sigma at each dt, then in dt.

    `band` keeps only points whose bias sits in the small bias regime. Outside
    it the bias saturates, and fitting through the saturated region would
    flatten every exponent toward zero.

    Errors in variables predicts a sigma exponent of exactly 2 for every
    weight. The dt exponent is predicted to differ by weight, because only
    alignment depends on velocity and therefore only alignment carries the
    term that goes as dt to the minus 3. See the spec's third addendum.
    """
    dts = sorted({r["dt"] for r in rows})
    sigmas = sorted({r["sigma"] for r in rows})
    per_dt = {}
    for dt in dts:
        kept = {"sigmas": [[], [], []], "bias": [[], [], []]}
        for s in sigmas:
            b = signed_bias(rows, dt, s)
            if b is None:
                continue
            for k in range(3):
                if band[0] < b[k] < band[1]:
                    kept["sigmas"][k].append(s)
                    kept["bias"][k].append(float(b[k]))
        exps = []
        coeffs = []
        for k in range(3):
            fit = _loglog_fit(kept["sigmas"][k], kept["bias"][k])
            if fit is None:
                exps.append(None)
                coeffs.append(None)
            else:
                exps.append(float(fit[0]))
                # the coefficient at sigma equal to 1, on a log scale
                coeffs.append(float(fit[1]))
        per_dt[dt] = {"sigma_exponent": exps, "log_coefficient": coeffs,
                      "n_points": [len(v) for v in kept["sigmas"]]}

    dt_exp = []
    dt_curv = []
    implied = []
    for k in range(3):
        xs = [dt for dt in dts if per_dt[dt]["log_coefficient"][k] is not None]
        ys = [per_dt[dt]["log_coefficient"][k] for dt in xs]
        if len(xs) < 2:
            dt_exp.append(None)
            dt_curv.append(None)
            implied.append(None)
            continue
        slope = float(np.polyfit(np.log(xs), ys, 1)[0])
        dt_exp.append(slope)
        # a pure power law is a straight line on these axes, so any quadratic
        # term is evidence for the two term form the derivation predicts
        if len(xs) >= 3:
            dt_curv.append(float(np.polyfit(np.log(xs), ys, 2)[0]))
        else:
            dt_curv.append(None)
        implied.append(-slope / 2.0)
    return {"per_dt": per_dt, "dt_exponent": dt_exp, "dt_curvature": dt_curv,
            "implied_threshold_exponent": implied, "band": list(band)}


def load_and_summarise(paths):
    """Merge shard files written by `sweep` and summarise the whole arm."""
    rows = []
    seconds = 0.0
    threshold = THRESHOLD
    for path in paths:
        with open(path) as f:
            payload = json.load(f)
        rows.extend(payload["rows"])
        seconds += payload["wall_clock_seconds"]
        threshold = payload["threshold"]
    merged = {"rows": rows, "threshold": threshold,
              "wall_clock_seconds": seconds, "shards": len(paths)}
    out = summarise(merged)
    out["n_fits"] = len(rows)
    out["cpu_seconds"] = seconds
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--t0", type=int, default=1)
    p.add_argument("--n-frames", type=int, default=None,
                   help="protocol 1: the same frame count at every timestep")
    p.add_argument("--duration", type=float, default=None,
                   help="protocol 2: the same simulated time at every timestep")
    p.add_argument("--sigmas", default=None,
                   help="comma separated; default is 12 log spaced 1e-5 to 1e-1")
    p.add_argument("--noise-seeds", default="0,1,2")
    p.add_argument("--true-velocities", action="store_true",
                   help="the control arm: noisy positions, exact velocities")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--label", default=None)
    p.add_argument("--stride", type=int, default=1,
                   help="thin the anchors: same frame count, wider span")
    p.add_argument("--only-dt", type=float, default=None,
                   help="restrict to runs at this timestep")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--skip-tripwires", action="store_true",
                   help="for timing calibration only, never for a result")
    args = p.parse_args()

    if args.n_frames is not None and args.duration is not None:
        raise SystemExit("choose one protocol: --n-frames or --duration")

    if args.sigmas:
        sigmas = [float(x) for x in args.sigmas.split(",")]
    else:
        sigmas = list(np.logspace(-5, -1, 12))
    seeds = [int(x) for x in args.noise_seeds.split(",")]

    payload = sweep(args.run_dir, sigmas, seeds, t0=args.t0,
                    n_frames=args.n_frames, duration=args.duration,
                    out=args.out, skip_tripwires=args.skip_tripwires,
                    true_velocities=args.true_velocities,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count, label=args.label,
                    stride=args.stride, only_dt=args.only_dt)
    if not args.quiet:
        print(json.dumps(summarise(payload), indent=2, sort_keys=True))
    print("shard %d of %d: wall clock %.1f s over %d fits"
          % (args.shard_index, args.shard_count,
             payload["wall_clock_seconds"], len(payload["rows"])))


if __name__ == "__main__":
    main()
