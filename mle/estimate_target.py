import numpy as np
from scipy.optimize import minimize
from parameters import params

def estimate_target(S_hover: np.ndarray, D_meas, params, x0= None) -> np.ndarray:
    """
    Continuous MLE-based 3D target localization using range measurements.

    Parameters
    ----------
    S_hover : ndarray, shape (K, 3)
        UAV hover positions [x_j, y_j, z_j]
    D_meas : ndarray, shape (K,)
        Measured distances
    params : dict
        Must contain:
            params["sim"]["L_x"], ["L_y"], ["L_z"]
            params["sim"]["sigma_0"]
    x0 : ndarray, shape (3,)
        Initial guess [x, y, z]

    Returns
    -------
    pos_hat : ndarray, shape (3,)
        Estimated target position
    """

    S_hover = np.asarray(S_hover, dtype=float)
    D_meas = np.asarray(D_meas, dtype=float)

    # -------------------------------------------------
    # Shape checks
    # -------------------------------------------------
    if S_hover.ndim != 2 or S_hover.shape[1] != 3:
        raise ValueError("S_hover must have shape (K, 3)")
    if D_meas.ndim != 1 or D_meas.shape[0] != S_hover.shape[0]:
        raise ValueError("D_meas must have shape (K,)")

    sim = params["sim"]
    Lx = sim["L_x"]
    Ly = sim["L_y"]
    Lz = sim["L_z"]
    sigma = sim["sigma_0"]

    # -------------------------------------------------
    # Negative log-likelihood (up to constant)
    # -------------------------------------------------
    def nll(theta):
        diff = S_hover - theta[None, :]   # (K,3)
        d_pred = np.linalg.norm(diff, axis=1)
        return np.sum((D_meas - d_pred) ** 2) / (sigma ** 2)

    # -------------------------------------------------
    # Box constraints
    # -------------------------------------------------
    bounds = [
        (0.0, Lx),
        (0.0, Ly),
        (0.0, Lz),
    ]


    # -------------------------------------------------
    # Initial guess
    # -------------------------------------------------
    if x0 is None:
        # geometric center of UAV positions (robust default)
        x0 = np.mean(S_hover, axis=0)

    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (3,):
        raise ValueError("x0 must have shape (3,)")

    # -------------------------------------------------
    # Optimization
    # -------------------------------------------------
    res = minimize(
        nll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not res.success:
        print("Warning: MLE did not fully converge:", res.message)

    return res.x
