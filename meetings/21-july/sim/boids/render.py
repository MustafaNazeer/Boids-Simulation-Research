import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np

def _draw_frame(ax, world):
    ax.clear()
    ax.set_xlim(0, world.world_size)
    ax.set_ylim(0, world.world_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if world.obstacles is not None:
        centers, radii = world.obstacles
        for c, r in zip(centers, radii):
            ax.add_patch(Circle(c, r, color="0.6", zorder=1))
    n = world.positions.shape[0]
    pre = getattr(world, "preplanned_mask", None)
    pre = np.zeros(n, dtype=bool) if pre is None else np.asarray(pre, dtype=bool)
    lead = getattr(world, "leader_mask", None)
    lead = np.zeros(n, dtype=bool) if lead is None else np.asarray(lead, dtype=bool)
    plain = ~(pre | lead)
    ax.scatter(world.positions[plain, 0], world.positions[plain, 1],
               s=7, c="steelblue", alpha=0.85, zorder=2)
    if pre.any():
        ax.scatter(world.positions[pre, 0], world.positions[pre, 1],
                   s=22, c="darkorange", alpha=0.95, zorder=3)
    if lead.any():
        ax.scatter(world.positions[lead, 0], world.positions[lead, 1],
                   s=45, c="gold", marker="^", alpha=0.95, zorder=4)
    if world.predator is not None:
        ax.scatter([world.predator["pos"][0]], [world.predator["pos"][1]],
                   s=60, c="crimson", marker="*", zorder=3)

def save_animation(world, frames, dt, out_path, fps=20):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(_):
        world.step(dt)
        _draw_frame(ax, world)

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def save_snapshot(world, out_path, zoom=False, pad=2.0):
    fig, ax = plt.subplots(figsize=(6, 6))
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
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
