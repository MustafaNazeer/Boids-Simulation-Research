import numpy as np

FEATURE_DIM = 5
FEATURE_DIM_NO_POSITION = 3


def feature_dim(include_position=True):
    return FEATURE_DIM if include_position else FEATURE_DIM_NO_POSITION


def node_features(positions, velocities, t, k_window, include_position=True):
    """Per agent motion history.

    With `include_position=True` the features are velocity x, velocity y,
    speed, and centroid relative position x and y.

    `include_position=False` drops the two positional channels, leaving
    velocity and speed only. This is what a genuine distance withheld
    ablation requires. Merely dropping the edge level distance feature is not
    enough: centroid relative positions let the head recover the pairwise
    distance from the difference of two node embeddings, so the model can
    reconstruct exactly the quantity the ablation is meant to deny it.
    """
    start = t - k_window + 1
    if start < 0:
        raise ValueError(
            "window of %d frames does not fit ending at t=%d" % (k_window, t))

    win_pos = positions[start:t + 1]
    win_vel = velocities[start:t + 1]

    # the centroid is recomputed per frame, so a constant offset applied to
    # every position cancels and the features stay translation invariant.
    # this is required by the unbounded `none` bounds mode, where the flock
    # drifts arbitrarily far from where it spawned
    speed = np.linalg.norm(win_vel, axis=-1, keepdims=True)
    channels = [win_vel, speed]

    if include_position:
        centroid = win_pos.mean(axis=1, keepdims=True)
        channels.append(win_pos - centroid)

    stacked = np.concatenate(channels, axis=-1)
    return np.transpose(stacked, (1, 0, 2)).astype(np.float32)
