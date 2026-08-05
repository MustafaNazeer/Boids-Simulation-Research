import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.boids.world import World, default_params
from sim.boids import metrics, render, preplanned as preplanned_mod
from sim.boids import config as config_mod
from sim.boids.recorder import TrajectoryRecorder

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "deliverables"))

def build_world(seed=7, n_boids=120, world_size=60.0, with_predator=True):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, world_size, size=(n_boids, 2))
    velocities = rng.uniform(-1, 1, size=(n_boids, 2))
    obstacles = (np.array([[18.0, 38.0], [44.0, 20.0]]), np.array([5.0, 5.0]))
    predator_state = {"pos": np.array([55.0, 55.0])} if with_predator else None
    return World(positions, velocities, default_params(), world_size,
                 obstacles=obstacles, predator_state=predator_state)

def build_flock_with_preplanned(seed, n_boids, world_size, n_preplanned, frames):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, world_size, size=(n_boids, 2))
    velocities = rng.uniform(-1, 1, size=(n_boids, 2))
    params = default_params()
    center = np.array([world_size / 2.0, world_size / 2.0])
    amplitude = (world_size / 2.0) * 0.8

    mask = np.zeros(n_boids, dtype=bool)
    agents = None
    if n_preplanned > 0:
        pre_ids = np.sort(rng.choice(n_boids, size=n_preplanned, replace=False))
        mask[pre_ids] = True
        plans = preplanned_mod.PLAN_CATALOG[:n_preplanned]
        agents = preplanned_mod.PreplannedAgents(
            plans, frames, amplitude, center, params["preplanned_speed"])
        pp_pos, pp_vel = agents.state_at(0)
        positions[pre_ids] = pp_pos
        velocities[pre_ids] = pp_vel
    return World(positions, velocities, params, world_size,
                 preplanned=agents, preplanned_mask=mask)

def build_leader_follower_world(seed=11, n_boids=120, world_size=60.0,
                                n_leaders=3, params=None):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, world_size, size=(n_boids, 2))
    velocities = rng.uniform(-1, 1, size=(n_boids, 2))
    if params is None:
        params = default_params()
    mask = np.zeros(n_boids, dtype=bool)
    if n_leaders > 0:
        mask[np.sort(rng.choice(n_boids, size=n_leaders, replace=False))] = True
    return World(positions, velocities, params, world_size,
                 leader_mask=mask, seed=seed)

def build_foraging_world(seed=11, n_boids=120, world_size=60.0, n_food=3,
                         bounds_mode="reflect", params=None):
    # the deck scenario: food plus the three boids rules only, so the
    # convergence, depletion, and re targeting cycle reads clearly
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, world_size, size=(n_boids, 2))
    velocities = rng.uniform(-1, 1, size=(n_boids, 2))
    if params is None:
        params = default_params()
    food_positions = rng.uniform(0, world_size, size=(n_food, 2))
    food_amounts = np.full(n_food, float(params["food_amount"]))
    return World(positions, velocities, params, world_size,
                 bounds_mode=bounds_mode,
                 food_positions=food_positions, food_amounts=food_amounts,
                 seed=seed)

def build_world_from_config(cfg):
    world_size = cfg["world"]["size"]
    n_boids = cfg["world"]["n_boids"]
    seed = cfg["world"]["seed"]
    frames = cfg["motion"]["frames"]
    params = config_mod.to_params(cfg)
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, world_size, size=(n_boids, 2))
    velocities = rng.uniform(-1, 1, size=(n_boids, 2))

    obstacles = None
    if cfg["scenario"]["obstacles"]:
        obstacles = (np.array([[world_size * 0.30, world_size * 0.63],
                               [world_size * 0.73, world_size * 0.33]]),
                     np.array([world_size * 0.08, world_size * 0.08]))
    predator_state = None
    if cfg["scenario"]["predator"]:
        predator_state = {"pos": np.array([world_size * 0.92,
                                           world_size * 0.92])}

    pre_mask = np.zeros(n_boids, dtype=bool)
    agents = None
    n_pre = cfg["preplanned"]["count"]
    if n_pre > 0:
        pre_ids = np.sort(rng.choice(n_boids, size=n_pre, replace=False))
        pre_mask[pre_ids] = True
        center = np.array([world_size / 2.0, world_size / 2.0])
        agents = preplanned_mod.PreplannedAgents(
            preplanned_mod.PLAN_CATALOG[:n_pre], frames,
            (world_size / 2.0) * 0.8, center, params["preplanned_speed"])
        pp_pos, pp_vel = agents.state_at(0)
        positions[pre_ids] = pp_pos
        velocities[pre_ids] = pp_vel

    # a preplanned agent is welded to its curve, so it can never also be a
    # wander leader; leaders are drawn from the boids left over
    leader_mask = np.zeros(n_boids, dtype=bool)
    free = np.where(~pre_mask)[0]
    n_leaders = min(cfg["leaders"]["count"], free.size)
    if n_leaders > 0:
        leader_mask[rng.choice(free, size=n_leaders, replace=False)] = True

    food_positions = None
    food_amounts = None
    n_food = cfg["food"]["count"]
    if n_food > 0:
        food_positions = rng.uniform(0, world_size, size=(n_food, 2))
        food_amounts = np.full(n_food, float(cfg["food"]["amount"]))

    return World(positions, velocities, params, world_size,
                 obstacles=obstacles, predator_state=predator_state,
                 preplanned=agents, preplanned_mask=pre_mask,
                 bounds_mode=cfg["world"]["bounds"],
                 leader_mask=leader_mask, seed=seed,
                 food_positions=food_positions, food_amounts=food_amounts)

def record_trajectory(world, frames, dt, out_path):
    recorder = TrajectoryRecorder(dt)
    for _ in range(frames):
        recorder.record(world.step_index, world)
        world.step(dt)
    recorder.to_csv(out_path)

def build_preplanned_debug_world(world_size, frames):
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
    parser = argparse.ArgumentParser(
        description="Generate the simulation deliverables. Behavior is read "
                    "from the YAML config, not hardcoded here.")
    parser.add_argument("--config", default=None,
                        help="path to a config file (default: sim/config.yaml)")
    args = parser.parse_args()
    cfg = config_mod.load(args.config)
    dt = cfg["motion"]["dt"]
    frames = cfg["motion"]["frames"]

    os.makedirs(OUT, exist_ok=True)

    world = build_world_from_config(cfg)
    render.save_animation(world, frames=frames, dt=dt,
                          out_path=os.path.join(OUT, "flock_escape.gif"))

    world = build_world_from_config(cfg)
    pol, nn, clusters = record_metrics(world, frames=frames, dt=dt)
    plot_series(pol, "polarization", "Flock polarization over time",
                os.path.join(OUT, "polarization.png"))
    plot_series(nn, "mean NN distance", "Spacing over time (spikes on escape)",
                os.path.join(OUT, "nn_distance.png"))
    plot_series(clusters, "cluster count", "Cluster count (rises on split)",
                os.path.join(OUT, "cluster_count.png"))

    world = build_world_from_config(cfg)
    for _ in range(40):
        world.step(dt)
    render.save_snapshot(world, os.path.join(OUT, "snapshot_early.png"))
    for _ in range(80):
        world.step(dt)
    render.save_snapshot(world, os.path.join(OUT, "snapshot_mid.png"))
    for _ in range(120):
        world.step(dt)
    render.save_snapshot(world, os.path.join(OUT, "snapshot_late.png"))

    world = build_world_from_config(cfg)
    record_trajectory(world, frames=frames, dt=dt,
                      out_path=os.path.join(OUT, "trajectory.csv"))

    lf = build_leader_follower_world()
    render.save_animation(lf, frames=300, dt=0.2,
                          out_path=os.path.join(OUT, "leader_follower.gif"))

    # by step 126 every follower has attached and all three leaders have a
    # group, while the flock is still spread widely enough to read; earlier
    # than that a large unattached cluster is still drifting
    lf = build_leader_follower_world()
    for _ in range(126):
        lf.step(0.2)
    render.save_snapshot(lf, os.path.join(OUT, "snapshot_leader_follower.png"),
                         zoom=True)

    lf = build_leader_follower_world()
    record_trajectory(lf, frames=300, dt=0.2,
                      out_path=os.path.join(OUT, "leader_follower.csv"))

    debug = build_preplanned_debug_world(world_size=60.0, frames=240)
    record_trajectory(debug, frames=240, dt=0.2,
                      out_path=os.path.join(OUT, "preplanned_debug.csv"))
    debug_anim = build_preplanned_debug_world(world_size=60.0, frames=240)
    render.save_animation(debug_anim, frames=240, dt=0.2,
                          out_path=os.path.join(OUT, "preplanned_debug.gif"))
    debug_for_plot = build_preplanned_debug_world(world_size=60.0, frames=240)
    save_preplanned_path_plot(debug_for_plot.preplanned, 60.0,
                              os.path.join(OUT, "snapshot_preplanned.png"))

    fw = build_foraging_world()
    render.save_animation(fw, frames=300, dt=0.2,
                          out_path=os.path.join(OUT, "foraging.gif"))

    fw = build_foraging_world()
    for _ in range(150):
        fw.step(0.2)
    render.save_snapshot(fw, os.path.join(OUT, "snapshot_foraging.png"))

    fw = build_foraging_world()
    record_trajectory(fw, frames=300, dt=0.2,
                      out_path=os.path.join(OUT, "foraging.csv"))

    iw = build_foraging_world(bounds_mode="none")
    render.save_animation(iw, frames=300, dt=0.2,
                          out_path=os.path.join(OUT, "infinite_world.gif"))

    print("Wrote deliverables to", OUT)

if __name__ == "__main__":
    main()
