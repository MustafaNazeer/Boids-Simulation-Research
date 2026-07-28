import numpy as np
from sim.boids import rules, predator as predator_mod, leader as leader_mod

def default_params():
    return {
        "separation_radius": 7.0,
        "alignment_radius": 16.0,
        "cohesion_radius": 16.0,
        "max_speed": 5.0,
        "min_speed": 1.5,
        "preplanned_speed": 5.0,
        "obstacle_avoid_range": 4.0,
        "fear_radius": 15.0,
        "leader_follow_radius": 20.0,
        "leader_wander_strength": 0.15,
        "weights": {
            "separation": 8.0,
            "alignment": 1.6,
            "cohesion": 0.5,
            "obstacle": 2.5,
            "flee": 3.5,
            "follow": 1.0,
            "wander": 2.0,
        },
    }

class World:

    def __init__(self, positions, velocities, params, world_size,
                 obstacles=None, predator_state=None,
                 preplanned=None, preplanned_mask=None, bounds_mode="reflect",
                 leader_mask=None, seed=None):
        self.positions = np.asarray(positions, dtype=float).copy()
        self.velocities = np.asarray(velocities, dtype=float).copy()
        self.params = params
        self.world_size = float(world_size)
        self.obstacles = obstacles
        self.predator = predator_state
        self.preplanned = preplanned
        self.preplanned_mask = preplanned_mask
        self.bounds_mode = bounds_mode
        self.step_index = 0
        self.time = 0.0
        n = self.positions.shape[0]
        self.rng = np.random.default_rng(seed)
        self.leader_mask = (None if leader_mask is None
                            else np.asarray(leader_mask, dtype=bool))
        self.leader_ids = np.full(n, leader_mod.NO_LEADER, dtype=int)
        if self.leader_mask is not None and self.leader_mask.any():
            self.leader_headings = leader_mod.initial_headings(n, self.rng)
        else:
            self.leader_headings = np.zeros(n)

    def _apply_bounds(self, positions, velocities):
        if self.bounds_mode == "wrap":
            return np.mod(positions, self.world_size), velocities
        return rules.reflect_bounds(positions, velocities, self.world_size)

    def _enabled(self, name):
        # a params dict built by hand (default_params) has no enabled map, so
        # every rule stays on unless a config explicitly switches it off
        return self.params.get("enabled", {}).get(name, True)

    def _reflect_headings(self, moved):
        # a bounced leader keeps its old wander heading unless it is mirrored
        # too, which otherwise steers it straight back into the wall every step
        # and pins the whole trailing group against the boundary
        if self.leader_mask is None or not self.leader_mask.any():
            return
        if self.bounds_mode == "wrap":
            return
        flip_x = (moved[:, 0] < 0.0) | (moved[:, 0] > self.world_size)
        flip_y = (moved[:, 1] < 0.0) | (moved[:, 1] > self.world_size)
        h = self.leader_headings
        h = np.where(flip_x, np.pi - h, h)
        h = np.where(flip_y, -h, h)
        self.leader_headings = h

    def _boid_acceleration(self):
        p = self.positions
        v = self.velocities
        prm = self.params
        w = prm["weights"]
        accels = []
        weights = []
        if self._enabled("separation"):
            accels.append(rules.separation(p, prm["separation_radius"]))
            weights.append(w["separation"])
        if self._enabled("alignment"):
            accels.append(rules.alignment(p, v, prm["alignment_radius"]))
            weights.append(w["alignment"])
        if self._enabled("cohesion"):
            accels.append(rules.cohesion(p, prm["cohesion_radius"]))
            weights.append(w["cohesion"])
        if self.obstacles is not None and self._enabled("obstacle_avoidance"):
            centers, radii = self.obstacles
            accels.append(rules.obstacle_avoidance(
                p, centers, radii, prm["obstacle_avoid_range"]))
            weights.append(w["obstacle"])
        if self.predator is not None and self._enabled("predator_flee"):
            accels.append(predator_mod.flee(
                p, self.predator["pos"], prm["fear_radius"]))
            weights.append(w["flee"])
        if self.leader_mask is not None and self.leader_mask.any():
            if self._enabled("leader_follow"):
                follow_accel, self.leader_ids = leader_mod.follow(
                    p, self.leader_mask, prm["leader_follow_radius"])
                accels.append(follow_accel)
                weights.append(w["follow"])
            if self._enabled("leader_wander"):
                wander_accel, self.leader_headings = leader_mod.wander(
                    self.leader_headings, self.leader_mask, self.rng,
                    prm["leader_wander_strength"])
                accels.append(wander_accel)
                weights.append(w["wander"])
        if not accels:
            return np.zeros_like(self.velocities)
        return rules.combine(accels, weights)

    def step(self, dt):
        accel = self._boid_acceleration()
        self.velocities = self.velocities + accel * dt
        self.velocities = rules.clamp_speed(self.velocities, self.params["max_speed"])
        self.velocities = rules.enforce_min_speed(self.velocities, self.params["min_speed"])
        moved = self.positions + self.velocities * dt
        self.positions, self.velocities = self._apply_bounds(
            moved, self.velocities)
        self._reflect_headings(moved)
        if self.preplanned is not None:
            idx = np.where(self.preplanned_mask)[0]
            pp_pos, pp_vel = self.preplanned.state_at(self.step_index + 1)
            self.positions[idx] = pp_pos
            self.velocities[idx] = pp_vel
        if self.predator is not None:
            pv = predator_mod.pursue(
                self.predator["pos"], self.positions,
                self.params["max_speed"] * 1.05)
            q = (self.predator["pos"] + pv * dt).reshape(1, -1)
            q, _ = self._apply_bounds(q, pv.reshape(1, -1))
            self.predator["pos"] = q[0]
        self.step_index += 1
        self.time = self.step_index * dt
