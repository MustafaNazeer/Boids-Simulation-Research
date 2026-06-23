import numpy as np
from sim.boids import rules, predator as predator_mod


def default_params():
    """Starting weights and radii. Expect to tune these for coherent flocking."""
    return {
        "perception_radius": 16.0,
        "separation_radius": 7.0,
        "max_speed": 3.0,
        "obstacle_avoid_range": 4.0,
        "fear_radius": 15.0,
        "weights": {
            "separation": 8.0,
            "alignment": 1.6,
            "cohesion": 0.5,
            "obstacle": 2.5,
            "flee": 3.5,
        },
    }


class World:
    """Holds boid state and advances it by one step at a time.

    obstacles: optional (centers (M,2) array, radii (M,) array) tuple.
    predator_state: optional dict with key "pos" (2,) array; when present the
        predator pursues the flock and boids flee from it.
    """

    def __init__(self, positions, velocities, params, world_size,
                 obstacles=None, predator_state=None):
        self.positions = np.asarray(positions, dtype=float).copy()
        self.velocities = np.asarray(velocities, dtype=float).copy()
        self.params = params
        self.world_size = float(world_size)
        self.obstacles = obstacles  # (centers, radii) or None
        self.predator = predator_state  # {"pos": (2,)} or None

    def _boid_acceleration(self):
        p = self.positions
        v = self.velocities
        prm = self.params
        w = prm["weights"]
        accels = [
            rules.separation(p, prm["separation_radius"]),
            rules.alignment(p, v, prm["perception_radius"]),
            rules.cohesion(p, prm["perception_radius"]),
        ]
        weights = [w["separation"], w["alignment"], w["cohesion"]]
        if self.obstacles is not None:
            centers, radii = self.obstacles
            accels.append(rules.obstacle_avoidance(
                p, centers, radii, prm["obstacle_avoid_range"]))
            weights.append(w["obstacle"])
        if self.predator is not None:
            accels.append(predator_mod.flee(
                p, self.predator["pos"], prm["fear_radius"]))
            weights.append(w["flee"])
        return rules.combine(accels, weights)

    def step(self, dt):
        accel = self._boid_acceleration()
        self.velocities = self.velocities + accel * dt
        self.velocities = rules.clamp_speed(self.velocities, self.params["max_speed"])
        self.positions = self.positions + self.velocities * dt
        self.positions = np.mod(self.positions, self.world_size)
        if self.predator is not None:
            pv = predator_mod.pursue(
                self.predator["pos"], self.positions,
                self.params["max_speed"] * 1.05)
            self.predator["pos"] = np.mod(
                self.predator["pos"] + pv * dt, self.world_size)
