# Boid Simulation

A pure Python boids flocking simulation built for undergraduate research on
emergent behavior in dynamic multi agent systems. The longer term goal is to
learn these local interaction rules with a Graph Neural Network.

Work is organized by research meeting date. Each meeting folder holds that
week's simulation source and a short slide deck describing what the
deliverables do.

## Layout

```
meetings/<date>/sim/           the simulation source (boids rules, predator,
                               metrics, world, rendering, and runners)
meetings/<date>/presentation/  the progress deck (progress-deck.pptx)
```

Generated artifacts (animations, metric plots, snapshots, and datasets) are
written into a local `deliverables/` folder when you run the code and are not
checked into the repository.

## Instructions

Run every command from inside the meeting folder you want, so the `sim` package
resolves. Each meeting is self contained.

### June 9

From `meetings/9-june`:

```
python3 -m sim.boids.live    # live animated window: flock, obstacles, predator
python3 -m sim.boids.run     # writes the animation, metric plots, and snapshots to deliverables/
```

### June 19

From `meetings/19-june`:

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
the progress deck in `meetings/26-june/presentation/`.

### June 30

June 30 delivered a batch data generator that turns the simulation into a
parameter diverse dataset for training the Graph Neural Network. It reruns the
simulation many times, randomizing the physics parameters within set ranges
from a single seed, and saves every run in two forms plus a manifest.

From `meetings/30-june`:

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

From `meetings/21-july`:

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

## Dependencies

Python 3, numpy, scipy, matplotlib, and Pillow (with ImageTk for the live
window). The July 21 configuration file also needs PyYAML.

## Note

The model is built and validated in pure Python. NVIDIA Isaac Sim is a planned
next step on a machine with a GPU.
