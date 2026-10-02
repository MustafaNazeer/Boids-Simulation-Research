"""The process a candidate actually runs in. Prints one JSON line."""

import json
import resource
import sys

SENTINEL = "@@DISCOVER_RESULT@@"

ADDRESS_SPACE = 2 * 1024 ** 3


def main(argv):
    cand_path, fit_path, val_path = argv
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE, ADDRESS_SPACE))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    try:
        from discover import baseline, data, verify
        with open(cand_path) as f:
            source = f.read()
        namespace = {}
        exec(compile(source, "candidate", "exec"), namespace)
        fit_windows = data.load_windows(fit_path)
        # Distance parameters are searched no lower than the data's nearest
        # neighbor scale, the same lower bound the baseline radii use.
        floor = baseline.radius_range(fit_windows)[0]
        fit = verify.evaluate(fit_windows,
                              data.load_windows(val_path),
                              namespace["features"],
                              namespace.get("PARAMS", {}),
                              radius_floor=floor)
        out = {"ok": True, "radius_floor": floor}
        out.update(vars(fit))
    except BaseException as exc:
        out = {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout = real_stdout
    print("\n" + SENTINEL + json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
