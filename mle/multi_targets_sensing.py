import numpy as np

def sense_two_targets(s_hover, P, sigma):
    """
    Simulate UAV sensing two static targets with unlabeled range measurements.

    Parameters
    ----------
    s_hover : ndarray, shape (N, 3)
        UAV hover positions
    P : ndarray, shape (3, 2)
        True target positions [p1, p2]
    sigma : float
        Measurement noise std

    Returns
    -------
    d_echoes : ndarray, shape (2, N)
        Unlabeled range echoes
    """
    s_hover = np.asarray(s_hover, dtype=float)
    P = np.asarray(P, dtype=float)

    assert s_hover.ndim == 2 and s_hover.shape[1] == 3
    assert P.shape == (3, 2)

    N = s_hover.shape[0]
    d_echoes = np.zeros((2, N))

    p1 = P[:, 0]
    p2 = P[:, 1]

    for j in range(N):
        s = s_hover[j]

        d1 = np.linalg.norm(s - p1) + sigma * np.random.randn()
        d2 = np.linalg.norm(s - p2) + sigma * np.random.randn()

        # random permutation (unlabeled echoes)
        if np.random.rand() < 0.5:
            d_echoes[:, j] = [d1, d2]
        else:
            d_echoes[:, j] = [d2, d1]

    return d_echoes


def associate_measurements(s_hover, d_echoes, P_init):
    """
    Hard association of range measurements to two targets
    using MLE criterion.

    Parameters
    ----------
    s_hover : ndarray, shape (N, 3)
        UAV hover positions
    d_echoes : ndarray, shape (2, N)
        Unlabeled range measurements
    P_init : ndarray, shape (3, 2)
        Initial estimates of target positions

    Returns
    -------
    D_assoc : ndarray, shape (2, N)
        Associated measurements:
        row 0 -> target 1
        row 1 -> target 2
    """
    s_hover = np.asarray(s_hover, dtype=float)
    d_echoes = np.asarray(d_echoes, dtype=float)
    P_init = np.asarray(P_init, dtype=float)

    assert s_hover.ndim == 2 and s_hover.shape[1] == 3
    assert d_echoes.shape[0] == 2
    assert P_init.shape == (3, 2)

    N = s_hover.shape[0]
    D_assoc = np.zeros((2, N))

    p1_init = P_init[:, 0]
    p2_init = P_init[:, 1]

    for j in range(N):
        s = s_hover[j]

        d1_hat = np.linalg.norm(s - p1_init)
        d2_hat = np.linalg.norm(s - p2_init)

        e0, e1 = d_echoes[:, j]

        # two possible assignments
        cost_A = (e0 - d1_hat) ** 2 + (e1 - d2_hat) ** 2
        cost_B = (e1 - d1_hat) ** 2 + (e0 - d2_hat) ** 2

        if cost_A <= cost_B:
            D_assoc[:, j] = [e0, e1]
        else:
            D_assoc[:, j] = [e1, e0]

    return D_assoc
