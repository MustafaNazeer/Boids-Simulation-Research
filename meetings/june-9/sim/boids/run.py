"""Run the boids simulation and write deliverables (animation, plots, snapshots).

Usage: python3 -m sim.boids.run  (run from the meetings/june-9 folder)
Outputs land in meetings/june-9/deliverables/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.boids.world import World, default_params
from sim.boids import metrics, render

HERE = os.path.dirname(os.path.abspath(__file__))
# run.py lives in sim/boids/, so deliverables/ is two levels up (meetings/june-9/).
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "deliverables"))


def build_world(seed=7, n_boids=120, world_size=60.0, with_predator=True):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, world_size, size=(n_boids, 2))
    velocities = rng.uniform(-1, 1, size=(n_boids, 2))
    obstacles = (np.array([[18.0, 38.0], [44.0, 20.0]]), np.array([5.0, 5.0]))
    predator_state = {"pos": np.array([55.0, 55.0])} if with_predator else None
    return World(positions, velocities, default_params(), world_size,
                 obstacles=obstacles, predator_state=predator_state)


def record_metrics(world, frames, dt, cluster_threshold=5.0):
    pol, nn, clusters = [], [], []
    for _ in range(frames):
        world.step(dt)
        pol.append(metrics.polarization(world.velocities))
        nn.append(metrics.mean_nearest_neighbor_distance(world.positions))
        clusters.append(metrics.cluster_count(world.positions, cluster_threshold))
    return np.array(pol), np.array(nn), np.array(clusters)


def plot_series(series, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(series, color="steelblue")
    ax.set_xlabel("frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) Animation of flock + obstacles + predator escape
    world = build_world()
    render.save_animation(world, frames=300, dt=0.2,
                          out_path=os.path.join(OUT, "flock_escape.gif"))

    # 2) Metrics over a fresh run with the same seed/config
    world = build_world()
    pol, nn, clusters = record_metrics(world, frames=300, dt=0.2)
    plot_series(pol, "polarization", "Flock polarization over time",
                os.path.join(OUT, "polarization.png"))
    plot_series(nn, "mean NN distance", "Spacing over time (spikes on escape)",
                os.path.join(OUT, "nn_distance.png"))
    plot_series(clusters, "cluster count", "Cluster count (rises on split)",
                os.path.join(OUT, "cluster_count.png"))

    # 3) Snapshots before / during / after
    world = build_world()
    for _ in range(40):
        world.step(0.2)
    render.save_snapshot(world, os.path.join(OUT, "snapshot_early.png"))
    for _ in range(80):
        world.step(0.2)
    render.save_snapshot(world, os.path.join(OUT, "snapshot_mid.png"))
    for _ in range(120):
        world.step(0.2)
    render.save_snapshot(world, os.path.join(OUT, "snapshot_late.png"))

    print("Wrote deliverables to", OUT)


if __name__ == "__main__":
    main()
