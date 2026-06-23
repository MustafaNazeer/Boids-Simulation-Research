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

## Running the live demo

From a meeting folder (for example meetings/9-june):

```
python3 -m sim.boids.live
```

A window opens and the flock animates in real time, steering around obstacles
and scattering when a predator approaches. See
`presentation/run-live-demo.md` for details.

## Generating the saved artifacts

```
python3 -m sim.boids.run
```

This writes the animation, metric plots, and snapshots used in the slides.

## Dependencies

Python 3, numpy, scipy, matplotlib, and Pillow (with ImageTk for the live
window).

## Note

The model is built and validated in pure Python. NVIDIA Isaac Sim is a planned
next step on a machine with a GPU.
