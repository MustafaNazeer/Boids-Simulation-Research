import csv
import numpy as np
from sim.boids import rules

HEADER = ["step", "boid_id", "x", "y", "z", "vel_x", "vel_y", "vel_z",
          "pre_planned", "separation", "cohesion", "alignment"]

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
            ])

    def to_csv(self, out_path):
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(self.rows)
