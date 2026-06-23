import matplotlib

from sim.boids.run import build_world
from sim.boids.render import _draw_frame
from sim.boids import metrics

matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DT = 0.2
INTERVAL_MS = 50

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

    anim = FuncAnimation(
        fig, update, interval=INTERVAL_MS, blit=False, cache_frame_data=False
    )
    plt.show()
    return anim

if __name__ == "__main__":
    main()
