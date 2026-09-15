import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from matplotlib.collections import PolyCollection
import numpy as np

# palette matched to Joel Pampam's simulation (Pygame-Boids-Gnn), so the two
# versions read as the same project side by side
BACKGROUND = "#0F1423"        # (15, 20, 35)
BOID_COLOR = "#64C8FF"        # (100, 200, 255)
LEADER_COLOR = "#FFD23C"      # (255, 210, 60)
PREDATOR_COLOR = "#DC2828"    # (220, 40, 40)
OBSTACLE_COLOR = "#C85050"    # (200, 80, 80)
PREPLANNED_COLOR = "darkorange"
FOOD_COLOR = "#50DC78"

def _triangles(positions, velocities, size):
    # Joel's boid shape: tip one size ahead along the heading, rear corners
    # at 0.6 size, 2.5 radians either side
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    def corner(offset, scale):
        return positions + scale * size * np.stack(
            [np.cos(angles + offset), np.sin(angles + offset)], axis=1)
    return np.stack(
        [corner(0.0, 1.0), corner(2.5, 0.6), corner(-2.5, 0.6)], axis=1)

def _draw_frame(ax, world):
    ax.clear()
    ax.set_facecolor(BACKGROUND)
    if world.bounds_mode == "none":
        # follow camera: a fixed world_size wide window centered on the
        # flock centroid; pans but never zooms, so the gif cannot pump
        center = world.positions.mean(axis=0)
        half = world.world_size / 2.0
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
    else:
        ax.set_xlim(0, world.world_size)
        ax.set_ylim(0, world.world_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if world.obstacles is not None:
        centers, radii = world.obstacles
        for c, r in zip(centers, radii):
            ax.add_patch(Circle(c, r, color=OBSTACLE_COLOR, zorder=1))
    food_pos = getattr(world, "food_positions", None)
    if food_pos is not None:
        full = float(world.params["food_amount"])
        for fp, amount in zip(food_pos, world.food_amounts):
            r = 0.5 + 1.0 * (amount / full if full > 0 else 0.0)
            ax.add_patch(Circle(fp, r, color=FOOD_COLOR,
                                alpha=0.9, zorder=2))
    n = world.positions.shape[0]
    pre = getattr(world, "preplanned_mask", None)
    pre = np.zeros(n, dtype=bool) if pre is None else np.asarray(pre, dtype=bool)
    lead = getattr(world, "leader_mask", None)
    lead = np.zeros(n, dtype=bool) if lead is None else np.asarray(lead, dtype=bool)
    plain = ~(pre | lead)
    boid_size = world.world_size / 75.0
    leader_size = world.world_size / 55.0
    if plain.any():
        ax.add_collection(PolyCollection(
            _triangles(world.positions[plain], world.velocities[plain],
                       boid_size),
            facecolors=BOID_COLOR, alpha=0.9, zorder=3))
    if pre.any():
        ax.add_collection(PolyCollection(
            _triangles(world.positions[pre], world.velocities[pre],
                       leader_size),
            facecolors=PREPLANNED_COLOR, alpha=0.95, zorder=4))
    if lead.any():
        ax.add_collection(PolyCollection(
            _triangles(world.positions[lead], world.velocities[lead],
                       leader_size),
            facecolors=LEADER_COLOR, alpha=0.95, zorder=5))
    if world.predator is not None:
        pvel = world.predator.get("vel", np.array([1.0, 0.0]))
        ax.add_collection(PolyCollection(
            _triangles(world.predator["pos"].reshape(1, 2),
                       np.asarray(pvel, dtype=float).reshape(1, 2),
                       leader_size * 1.2),
            facecolors=PREDATOR_COLOR, zorder=6))

def save_animation(world, frames, dt, out_path, fps=20):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BACKGROUND)

    def update(_):
        world.step(dt)
        _draw_frame(ax, world)

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def save_snapshot(world, out_path, zoom=False, pad=2.0):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BACKGROUND)
    _draw_frame(ax, world)
    if zoom:
        # frame the agents instead of the whole world, for figures where the
        # flock is small relative to the world and would otherwise be a speck
        lo = world.positions.min(axis=0) - pad
        hi = world.positions.max(axis=0) + pad
        side = float(np.max(hi - lo))
        mid = (hi + lo) / 2.0
        ax.set_xlim(mid[0] - side / 2.0, mid[0] + side / 2.0)
        ax.set_ylim(mid[1] - side / 2.0, mid[1] + side / 2.0)
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
