import numpy as np

def relative_distance(S_hover, s_target):
    """
    3D relative distance between UAV hovering points and target.

    Parameters
    ----------
    S_hover : ndarray, shape (3, K)
        UAV hovering positions [x_k; y_k; z_k]
    s_target : array-like, shape (3,)
        Target position [x_t, y_t, z_t]

    Returns
    -------
    ds_vec : ndarray, shape (K,)
        Euclidean distances
    """
    S_hover = np.asarray(S_hover, dtype=float)
    s_target = np.asarray(s_target, dtype=float).reshape(3,)

    dx = S_hover[0, :] - s_target[0]
    dy = S_hover[1, :] - s_target[1]
    dz = S_hover[2, :] - s_target[2]

    ds_vec = np.sqrt(dx**2 + dy**2 + dz**2)
    return ds_vec