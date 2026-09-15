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
        # the title reports whichever behaviors the config actually switched
        # on, so a foraging run does not display a leader statistic that is
        # always zero because the run has no leaders
        parts = ["polarization {:.2f}".format(
            metrics.polarization(world.velocities))]
        if world.leader_mask is not None and world.leader_mask.any():
            parts.append("following {:.0%}".format(
                metrics.follower_attachment(world.leader_ids,
                                            world.leader_mask)))
        if world.food_positions is not None:
            parts.append("foraging {:.0%}".format(
                metrics.food_attachment(world.food_ids)))
        ax.set_title("live: " + ", ".join(parts))

    anim = FuncAnimation(
        fig, update, interval=INTERVAL_MS, blit=False, cache_frame_data=False
    )
    plt.show()
    return anim

if __name__ == "__main__":
    main()
