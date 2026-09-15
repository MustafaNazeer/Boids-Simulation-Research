import csv
import numpy as np
from sim.boids import rules, leader as leader_mod, food as food_mod

HEADER = ["step", "boid_id", "x", "y", "z", "vel_x", "vel_y", "vel_z",
          "pre_planned", "separation", "cohesion", "alignment",
          "is_leader", "leader_id", "food_id"]

class TrajectoryRecorder:

    def __init__(self, dt, z_height=5.0, include_self=False):
        self.dt = float(dt)
        self.z_height = float(z_height)
        self.include_self = bool(include_self)
        self.rows = []

    def record(self, step, world):
        pos = world.positions
        vel = world.velocities
        n = pos.shape[0]
        mask = world.preplanned_mask
        if mask is None:
            mask = np.zeros(n, dtype=bool)
        # the leader assignment is recomputed here, from the same positions the
        # other three graphs are built from, so every logged edge set belongs to
        # the state logged beside it rather than to the previous step
        lead_mask = getattr(world, "leader_mask", None)
        if lead_mask is None or not np.asarray(lead_mask, dtype=bool).any():
            lead_mask = np.zeros(n, dtype=bool)
            lead_ids = np.full(n, leader_mod.NO_LEADER, dtype=int)
        else:
            _, lead_ids = leader_mod.follow(
                pos, lead_mask, world.params["leader_follow_radius"])
        food_pos = getattr(world, "food_positions", None)
        if food_pos is None:
            food_ids = np.full(n, food_mod.NO_FOOD, dtype=int)
        else:
            # recomputed from the logged positions, like the leader edge, so
            # every stored edge set belongs to the state stored beside it
            _, food_ids = food_mod.seek(
                pos, food_pos, world.food_amounts,
                world.params["food_sensing_radius"])
        sep = rules.neighbor_id_lists(
            pos, world.params["separation_radius"], self.include_self)
        coh = rules.neighbor_id_lists(
            pos, world.params["cohesion_radius"], self.include_self)
        ali = rules.neighbor_id_lists(
            pos, world.params["alignment_radius"], self.include_self)
        for i in range(n):
            self.rows.append([
                int(step), i,
                pos[i, 0], pos[i, 1], self.z_height,
                vel[i, 0], vel[i, 1], 0.0,
                bool(mask[i]),
                "-".join(map(str, sep[i])),
                "-".join(map(str, coh[i])),
                "-".join(map(str, ali[i])),
                bool(lead_mask[i]),
                int(lead_ids[i]),
                int(food_ids[i]),
            ])

    def to_csv(self, out_path):
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(self.rows)
