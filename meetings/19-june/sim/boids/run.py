"""Run the boids simulation and write deliverables (animation, plots, snapshots).

Usage: python3 -m sim.boids.run  (run from the meetings/19-june folder)
Outputs land in meetings/19-june/deliverables/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.boids.world import World, default_params
from sim.boids import metrics, render, preplanned as preplanned_mod
from sim.boids.recorder import TrajectoryRecorder

HERE = os.path.dirname(os.path.abspath(__file__))
# run.py lives in sim/boids/, so deliverables/ is two levels up (meetings/19-june/).
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "deliverables"))


def build_world(seed=7, n_boids=120, world_size=60.0, with_predator=True):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, world_size, size=(n_boids, 2))
    velocities = rng.uniform(-1, 1, size=(n_boids, 2))
    obstacles = (np.array([[18.0, 38.0], [44.0, 20.0]]), np.array([5.0, 5.0]))
    predator_state = {"pos": np.array([55.0, 55.0])} if with_predator else None
    return World(positions, velocities, default_params(), world_size,
                 obstacles=obstacles, predator_state=predator_state)


def build_flock_with_leaders(seed, n_boids, world_size, n_leaders, frames):
    """A flock with n_leaders pre-planned leaders, no predator, for clean data.

    Leader positions are initialized to their path start so the first recorded
    frame already sits on the path.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, world_size, size=(n_boids, 2))
    velocities = rng.uniform(-1, 1, size=(n_boids, 2))
    params = default_params()
    center = np.array([world_size / 2.0, world_size / 2.0])
    amplitude = (world_size / 2.0) * 0.8

    mask = np.zeros(n_boids, dtype=bool)
    agents = None
    if n_leaders > 0:
        leader_ids = np.sort(rng.choice(n_boids, size=n_leaders, replace=False))
        mask[leader_ids] = True
        plans = preplanned_mod.PLAN_CATALOG[:n_leaders]
        agents = preplanned_mod.PreplannedAgents(
            plans, frames, amplitude, center, params["preplanned_speed"])
        pp_pos, pp_vel = agents.state_at(0)
        positions[leader_ids] = pp_pos
        velocities[leader_ids] = pp_vel
    return World(positions, velocities, params, world_size,
                 preplanned=agents, preplanned_mask=mask)


def record_trajectory(world, frames, dt, out_path):
    """Step the world, recording every frame, then write the CSV."""
    recorder = TrajectoryRecorder(dt)
    for _ in range(frames):
        recorder.record(world.step_index, world)
        world.step(dt)
    recorder.to_csv(out_path)


def build_preplanned_debug_world(world_size, frames):
    """All eight pre-planned plans, every agent pre-planned, for verification."""
    params = default_params()
    center = np.array([world_size / 2.0, world_size / 2.0])
    amplitude = (world_size / 2.0) * 0.8
    agents = preplanned_mod.PreplannedAgents(
        preplanned_mod.PLAN_CATALOG, frames, amplitude, center,
        params["preplanned_speed"])
    pos0, vel0 = agents.state_at(0)
    mask = np.ones(agents.count, dtype=bool)
    return World(pos0.copy(), vel0.copy(), params, world_size,
                 preplanned=agents, preplanned_mask=mask)


def save_preplanned_path_plot(agents, world_size, out_path):
    """Plot the full pre-planned paths (figure 8 and line shapes) to a PNG."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for path in agents.paths:
        ax.plot(path[:, 0], path[:, 1], color="goldenrod", linewidth=1.5, alpha=0.9)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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

    # 4) Trajectory CSV: clean flock with two pre-planned leaders, no predator
    flock = build_flock_with_leaders(seed=7, n_boids=120, world_size=60.0,
                                     n_leaders=2, frames=300)
    record_trajectory(flock, frames=300, dt=0.2,
                      out_path=os.path.join(OUT, "trajectory.csv"))

    # 5) Pre-planned debug scenario: known paths, CSV plus a gif and a snapshot
    debug = build_preplanned_debug_world(world_size=60.0, frames=240)
    record_trajectory(debug, frames=240, dt=0.2,
                      out_path=os.path.join(OUT, "preplanned_debug.csv"))
    debug_anim = build_preplanned_debug_world(world_size=60.0, frames=240)
    render.save_animation(debug_anim, frames=240, dt=0.2,
                          out_path=os.path.join(OUT, "preplanned_debug.gif"))
    debug_for_plot = build_preplanned_debug_world(world_size=60.0, frames=240)
    save_preplanned_path_plot(debug_for_plot.preplanned, 60.0,
                              os.path.join(OUT, "snapshot_preplanned.png"))

    print("Wrote deliverables to", OUT)


if __name__ == "__main__":
    main()
