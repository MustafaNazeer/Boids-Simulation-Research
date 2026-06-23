import matplotlib
matplotlib.use("Agg")  # headless, no display required
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
    mask = getattr(world, "preplanned_mask", None)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        normal = ~mask
        ax.scatter(world.positions[normal, 0], world.positions[normal, 1],
                   s=7, c="steelblue", alpha=0.85, zorder=2)
        ax.scatter(world.positions[mask, 0], world.positions[mask, 1],
                   s=22, c="gold", alpha=0.95, zorder=3)
    else:
        ax.scatter(world.positions[:, 0], world.positions[:, 1],
                   s=7, c="steelblue", alpha=0.85, zorder=2)
    if world.predator is not None:
        ax.scatter([world.predator["pos"][0]], [world.predator["pos"][1]],
                   s=60, c="crimson", marker="*", zorder=3)


def save_animation(world, frames, dt, out_path, fps=20):
    """Step the world `frames` times, writing an animated gif to out_path."""
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(_):
        world.step(dt)
        _draw_frame(ax, world)

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_snapshot(world, out_path):
    """Save a single PNG of the current world state."""
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_frame(ax, world)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
