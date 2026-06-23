"""Watch the boids simulation live in an interactive window.

Usage: python3 -m sim.boids.live  (run from the meetings/june-9 folder)
Requires a display and the TkAgg GUI backend. Close the window to stop.

This reuses the same World, configuration, and drawing code as the saved
animation. The only difference is that frames are shown live in a window
instead of being written to a gif.
"""
import matplotlib

# Import the modules that force the headless Agg backend first, then override
# to an interactive backend before any figure is created.
from sim.boids.run import build_world
from sim.boids.render import _draw_frame
from sim.boids import metrics

matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DT = 0.2
INTERVAL_MS = 50  # about 20 frames per second


def main():
    world = build_world()
    fig, ax = plt.subplots(figsize=(7, 7))

    def update(_):
        world.step(DT)
        _draw_frame(ax, world)
        pol = metrics.polarization(world.velocities)
        clusters = metrics.cluster_count(world.positions, 5.0)
        ax.set_title(
            "boids live: polarization {:.2f}, clusters {}".format(pol, clusters)
        )

    # Keep a reference to the animation so it is not garbage collected.
    anim = FuncAnimation(
        fig, update, interval=INTERVAL_MS, blit=False, cache_frame_data=False
    )
    plt.show()
    return anim


if __name__ == "__main__":
    main()
