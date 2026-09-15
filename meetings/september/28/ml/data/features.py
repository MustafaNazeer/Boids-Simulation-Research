import numpy as np

FEATURE_DIM = 5
FEATURE_DIM_NO_POSITION = 3


def feature_dim(include_position=True, n_type_channels=0):
    base = FEATURE_DIM if include_position else FEATURE_DIM_NO_POSITION
    return base + int(n_type_channels)


def node_features(positions, velocities, t, k_window, include_position=True,
                  agent_types=None):
    """Per agent motion history.

    With `include_position=True` the features are velocity x, velocity y,
    speed, and centroid relative position x and y.

    `agent_types`, when given, is an (n_boids, C) array appended as C extra
    channels. Agent type is a property of the agent rather than of the frame,
    so it is broadcast unchanged across the window. In this simulation the
    natural choice is two channels, is_leader and is_preplanned, which
    together cover about 4.5 percent of agents.

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

    if agent_types is not None:
        types = np.asarray(agent_types, dtype=np.float32)
        if types.ndim != 2 or types.shape[0] != win_vel.shape[1]:
            raise ValueError(
                "agent_types must be (n_boids, C), got %r for %d agents"
                % (types.shape, win_vel.shape[1]))
        # channels here are (window, n_boids, C), so broadcast the per agent
        # type across the time axis before concatenating
        channels.append(np.broadcast_to(
            types[None, :, :], (win_vel.shape[0],) + types.shape))

    stacked = np.concatenate(channels, axis=-1)
    return np.transpose(stacked, (1, 0, 2)).astype(np.float32)
