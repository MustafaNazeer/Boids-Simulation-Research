# Boid Simulation

A pure Python boids flocking simulation built for undergraduate research on
emergent behavior in dynamic multi agent systems. The longer term goal is to
learn these local interaction rules with a Graph Neural Network.

Work is organized by research meeting date. Each meeting folder holds that
week's simulation source and a short slide deck describing what the deliverables
do.

## Layout

```
meetings/<date>/sim/           the simulation source (boids rules, predator,
                               metrics, world, rendering, and runners)
meetings/<date>/presentation/  the progress slides and the deck generator
```

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

Build the progress deck from `meetings/19-june/presentation`:

```
python3 build_deck.py        # writes progress-deck.pptx
```

See `presentation/run-live-demo.md` inside a meeting folder for live demo notes.

## Dependencies

Python 3, numpy, scipy, matplotlib, and Pillow (with ImageTk for the live
window). Building the slide deck also needs python-pptx.

## Note

The model is built and validated in pure Python. NVIDIA Isaac Sim is a planned
next step on a machine with a GPU.
