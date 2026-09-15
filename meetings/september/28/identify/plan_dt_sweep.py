"""Plan the sampling rate sweep. Seed is recorded in RESULTS-sampling-rate.md.

Frame counts are the larger of what the two protocols need, so one generation
serves both: protocol 1 reads a 50 frame window at every timestep, protocol 2
reads a 10 second window, which is 200, 100, 50 and 25 frames.
"""
import os
import sys

from identify import manifest

SEED = 20260914
DTS = [0.05, 0.1, 0.2, 0.4]
RUNS_PER_DT = 5
FRAMES_FOR = {0.05: 210, 0.1: 110, 0.2: 60, 0.4: 60}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "identify/data_runs/dt_sweep"
    rows = manifest.plan(DTS, RUNS_PER_DT, FRAMES_FOR, SEED)
    path = manifest.write(rows, out)
    print("Planned %d runs into %s" % (len(rows), path))
    print("dt groups: %s" % {d: sum(1 for r in rows if r["dt"] == d)
                             for d in DTS})
    print("frames:    %s" % {r["dt"]: r["frames"] for r in rows})
    print("total simulated frames: %d"
          % sum(r["frames"] for r in rows))
