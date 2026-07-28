import argparse
import matplotlib

from sim.boids.run import build_world_from_config
from sim.boids.render import _draw_frame
from sim.boids import metrics, config as config_mod

matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

INTERVAL_MS = 50

def main():
    parser = argparse.ArgumentParser(
        description="Live animated window, configured from the YAML config.")
    parser.add_argument("--config", default=None,
                        help="path to a config file (default: sim/config.yaml)")
    args = parser.parse_args()
    cfg = config_mod.load(args.config)
    DT = cfg["motion"]["dt"]
    world = build_world_from_config(cfg)
    fig, ax = plt.subplots(figsize=(7, 7))

    def update(_):
        world.step(DT)
        _draw_frame(ax, world)
        pol = metrics.polarization(world.velocities)
        attached = metrics.follower_attachment(world.leader_ids,
                                               world.leader_mask)
        ax.set_title(
            "leader-follower live: polarization {:.2f}, attached {:.0%}".format(
                pol, attached)
        )

    anim = FuncAnimation(
        fig, update, interval=INTERVAL_MS, blit=False, cache_frame_data=False
    )
    plt.show()
    return anim

if __name__ == "__main__":
    main()
