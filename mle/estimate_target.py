# Trong hàm này, hệ thống thực hiện sử dụng các phép đo khoảng cách tới vật thể để ước lượng vị trí vật thể.
import numpy as np
from scipy.optimize import minimize


def estimate_target(S_hover, D_meas, params, x0):
    """
    Continuous MLE-based 3D target localization using range measurements.

    Parameters
    ----------
    S_hover : ndarray, shape (3, K)
        UAV hover positions [x_j; y_j; z_j]
    D_meas : ndarray, shape (K,)
        Measured distances
    params : dict
        Must contain:
            params["sim"]["L_x"], ["L_y"], ["L_z"]
            params["sim"]["sigma_0"]
    x0 : ndarray, optional, shape (3,)
        Initial guess [x, y, z]

    Returns
    -------
    pos_hat : ndarray, shape (3,)
        Estimated target position [x_hat; y_hat; z_hat]
    """

    S_hover = np.asarray(S_hover, dtype=float)
    D_meas = np.asarray(D_meas, dtype=float)

    if S_hover.shape[0] != 3:
        raise ValueError("S_hover must have shape (3, K)")

    sim = params["sim"]
    Lx = sim["L_x"]
    Ly = sim["L_y"]
    Lz = sim["L_z"]
    sigma = sim["sigma_0"]

    x_j = S_hover[0, :]
    y_j = S_hover[1, :]
    z_j = S_hover[2, :]

    # -------------------------------------------------
    # Negative log-likelihood (up to constant)
    # -------------------------------------------------
    def nll(theta):
        x_t, y_t, z_t = theta

        d_pred = np.sqrt(
            (x_j - x_t) ** 2
            + (y_j - y_t) ** 2
            + (z_j - z_t) ** 2
        )

        return np.sum((D_meas - d_pred) ** 2) / (sigma ** 2)


    # -------------------------------------------------
    # Box constraints (soft physical region)
    # -------------------------------------------------
    bounds = [
        (0.0, Lx),
        (0.0, Ly),
        (0.0, Lz),
    ]

    # -------------------------------------------------
    # Optimization
    # -------------------------------------------------
    res = minimize(
        nll,
        x0, #x0 chính là vị trí phỏng đoán ban đầu của UAV.
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not res.success:
        # fail-safe: still return best found
        print("Warning: MLE did not fully converge:", res.message)

    return res.x