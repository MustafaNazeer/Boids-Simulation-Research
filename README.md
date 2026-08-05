# Boid Simulation

A pure Python boids flocking simulation built for undergraduate research on
emergent behavior in dynamic multi agent systems. The longer term goal is to
learn these local interaction rules with a Graph Neural Network.

Work is organized by research meeting date. Each meeting folder holds that
week's simulation source and a short slide deck describing what the
deliverables do.

## Layout

```
meetings/<month>/<day>/sim/               the simulation source (boids rules, predator,
                               metrics, world, rendering, and runners)
meetings/<month>/<day>/presentation/  the progress deck (progress-deck.pptx)
```

Generated artifacts (animations, metric plots, snapshots, and datasets) are
written into a local `deliverables/` folder when you run the code and are not
checked into the repository.

## Instructions

Run every command from inside the meeting folder you want, so the `sim` package
resolves. Each meeting is self contained.

### June 9

From `meetings/june/09`:

```
python3 -m sim.boids.live    # live animated window: flock, obstacles, predator
python3 -m sim.boids.run     # writes the animation, metric plots, and snapshots to deliverables/
```

### June 19

From `meetings/june/19`:

```
python3 -m sim.boids.live    # live animated window: flock, obstacles, predator
python3 -m sim.boids.run     # writes the June 9 artifacts plus the trajectory data
```

In June 19, `python3 -m sim.boids.run` additionally writes `trajectory.csv`, a
standalone `preplanned_debug.csv`, `preplanned_debug.gif`, and
`snapshot_preplanned.png` (the preplanned figure 8 and line path shapes) into
`deliverables/`.

### June 26

June 26 covered the move to the UTA high performance computing cluster: getting
GPU access through Dr. Patino and learning the Slurm workflow that shares the
GPUs fairly. There is no new simulation code for this week. The deliverable is
the progress deck in `meetings/june/26/presentation/`.

### June 30

June 30 delivered a batch data generator that turns the simulation into a
parameter diverse dataset for training the Graph Neural Network. It reruns the
simulation many times, randomizing the physics parameters within set ranges
from a single seed, and saves every run in two forms plus a manifest.

From `meetings/june/30`:

```
python3 -m sim.boids.collect --runs 8 --frames 300   # writes the dataset to deliverables/dataset/
python3 -m sim.boids.live                            # live animated window
python3 -m sim.boids.run                             # single run: animation, plots, trajectory
```

`collect.py` accepts `--runs`, `--frames`, `--dt`, `--seed`, `--world-size`,
and `--out`. Each run writes a readable `run_XXXX.csv` (the per agent
trajectory) and a compact compressed `run_XXXX.npz` (positions, velocities, the
preplanned mask, and the run parameters) into `deliverables/dataset/`, together
with a `manifest.csv` that ties every run back to the exact parameters that
produced it. The whole dataset is reproducible from the single `--seed` value
under the same run settings.

### July 21

July 21 adds the Leader-Follower behavior and a configuration file.

A configurable number of leaders wander on a drifting heading while every
other boid steers toward its nearest leader within the follow radius. Leaders
and followers alike still run separation, alignment, and cohesion. The
recorder logs `is_leader` and `leader_id` alongside the three neighbor lists,
so the follower to leader link is stored as a fourth graph.

From `meetings/july/21`:

```
python3 -m sim.boids.live      # live animated window
python3 -m sim.boids.run       # writes the animations, plots, snapshots, and trajectories
python3 -m sim.boids.collect   # batch parameter sweep into deliverables/dataset/
```

All three entry points read `sim/config.yaml`, which holds every tunable value
and gives each rule its own radius and its own `enabled` flag. Setting a rule
to `enabled: false` removes it from the simulation entirely rather than
weighting it to zero. Point any entry point at a different file with
`--config <path>`:

```
python3 -m sim.boids.run --config my-experiment.yaml
```

A config that names an unknown key, an invalid bounds mode, or more leaders
than boids is rejected at load time with a message naming the offending key.

### July 28

July 28 adds food foraging and a truly infinite world.

A configurable number of depleting food sources sit in the world. A boid
steers toward the nearest non empty source once it falls inside the sensing
radius, drains that source by one unit per step while inside the much smaller
eat radius, and an exhausted source respawns at full amount at a new seeded
location. Separation, alignment, and cohesion all stay active alongside the
food seek term. The recorder logs a `food_id` column after `leader_id`, the
targeted source index or -1, recomputed from the logged positions each step so
the stored graph matches the stored state.

A third bounds mode, `none`, removes the world boundary entirely: positions
and velocities pass through unchanged, and the renderer switches to a follow
camera that pans a fixed size window centered on the flock without ever
zooming. Food respawns near the flock centroid in this mode so foraging keeps
working however far the flock travels.

From `meetings/july/28`:

```
python3 -m sim.boids.live      # live animated window
python3 -m sim.boids.run       # writes the animations, plots, snapshots, and trajectories
python3 -m sim.boids.collect   # batch parameter sweep into deliverables/dataset/
```

The shipped `sim/config.yaml` is the foraging scenario: three food sources
and obstacles, with no leaders and no predator, so the plain commands show
the new behavior without any extra arguments. Add leaders with
`leaders.count`, bring back the predator with `scenario.predator`, or switch
foraging off with `food.count: 0`. `sim/foraging.yaml` is the same scenario
with the obstacles removed as well.

Obstacles are enforced as a hard constraint rather than by steering alone.
Steering forces grow with distance, so a strong enough pull toward food or
toward the flock center could otherwise drag a boid straight through an
obstacle; anything that ends a step inside one is now placed back on the
surface and loses only the part of its velocity driving it deeper. Food is
never placed inside an obstacle either.

## Dependencies

Python 3, numpy, scipy, matplotlib, and Pillow (with ImageTk for the live
window). The configuration file, introduced July 21, also needs PyYAML.

## Note

The model is built and validated in pure Python. NVIDIA Isaac Sim is a planned
next step on a machine with a GPU.
