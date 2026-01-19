import numpy as np


def users_quad_distance(S, s_c):
    """
    Calculates the distance from UAV trajectory points to a communication user
    in 3D space.

    Parameters
    ----------
    S : np.ndarray, shape (3, N)
        UAV trajectory points:
        S[0, :] = x coordinates
        S[1, :] = y coordinates
        S[2, :] = z coordinates

    s_c : array-like, shape (3,)
        User position [x_user, y_user, z_user]

    Returns
    -------
    relative_distance : np.ndarray, shape (N,)
        Euclidean distances from UAV to user
    """

    S = np.asarray(S, dtype=float)
    s_c = np.asarray(s_c, dtype=float).reshape(3,)

    if S.shape[0] != 3:
        raise ValueError(f"S must have shape (3, N), got {S.shape}")

    # Relative position vectors
    dx = S[0, :] - s_c[0]
    dy = S[1, :] - s_c[1]
    dz = S[2, :] - s_c[2]

    relative_distance = np.sqrt(dx**2 + dy**2 + dz**2)
    return relative_distance