import numpy as np

def relative_distance(S_hov, s_target):
    """
    Compute distances between UAV hover points and a target.

    Parameters
    ----------
    S_hov : ndarray, shape (N, 3)
    s_target : ndarray, shape (3)

    Returns
    -------
    d : ndarray, shape (N)
    """
    S_hov = np.asarray(S_hov, dtype=float)
    s_target = np.asarray(s_target, dtype=float)

    return np.linalg.norm(S_hov - s_target, axis=1)
