import numpy as np


def sense_two_targets(S_hover, p1, p2, sigma):
    """
    Simulate UAV sensing two static targets with range measurements.

    Parameters
    ----------
    S_hover : ndarray, shape (3, N)
        UAV hover positions
    p1, p2 : ndarray, shape (3,)
        True target positions
    sigma : float
        Measurement noise std

    Returns
    -------
    D_echoes : ndarray, shape (2, N)
        Unlabeled range echoes (association unknown)
    """

    S_hover = np.asarray(S_hover, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    N = S_hover.shape[1]
    D_echoes = np.zeros((2, N))

    for j in range(N):
        s = S_hover[:, j]

        d1 = np.linalg.norm(s - p1) + sigma * np.random.randn()
        d2 = np.linalg.norm(s - p2) + sigma * np.random.randn()

        if np.random.rand() < 0.5:
            D_echoes[:, j] = [d1, d2]
        else:
            D_echoes[:, j] = [d2, d1]

    return D_echoes


def associate_measurements(S_hover, D_echoes, p1_init, p2_init):
    """
    Hard association of range measurements to two targets using MLE criterion.

    Parameters
    ----------
    S_hover : ndarray, shape (3, N)
    D_echoes : ndarray, shape (2, N)
    p1_init, p2_init : ndarray, shape (3,)
        Initial rough estimates of target positions

    Returns
    -------
    D1, D2 : ndarray, shape (N,)
        Measurements assigned to target 1 and 2
    """

    S_hover = np.asarray(S_hover, dtype=float)
    D_echoes = np.asarray(D_echoes, dtype=float)

    N = S_hover.shape[1]
    D1 = np.zeros(N)
    D2 = np.zeros(N)

    for j in range(N):
        s = S_hover[:, j]

        d1_hat = np.linalg.norm(s - p1_init)
        d2_hat = np.linalg.norm(s - p2_init)

        e0, e1 = D_echoes[:, j]

        cost_A = (e0 - d1_hat) ** 2 + (e1 - d2_hat) ** 2
        cost_B = (e1 - d1_hat) ** 2 + (e0 - d2_hat) ** 2

        if cost_A <= cost_B:
            D1[j], D2[j] = e0, e1
        else:
            D1[j], D2[j] = e1, e0

    return D1, D2
