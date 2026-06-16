import numpy as np


def flee(positions, predator_pos, fear_radius):
    """Boids within fear_radius steer directly away from the predator."""
    n, d = positions.shape
    accel = np.zeros((n, d))
    offset = positions - predator_pos
    dist = np.linalg.norm(offset, axis=1)
    mask = (dist < fear_radius) & (dist > 0)
    accel[mask] = offset[mask] / dist[mask, None]
    return accel


def pursue(predator_pos, positions, max_speed):
    """Return a velocity that heads the predator toward the nearest boid."""
    offset = positions - predator_pos
    dist = np.linalg.norm(offset, axis=1)
    nearest = int(np.argmin(dist))
    direction = offset[nearest]
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.zeros_like(predator_pos)
    return direction / norm * max_speed
