import numpy as np
from radar_sensing_model.relative_distance import relative_distance


def fisher_matrix_entry(S_hov, s_target, entry, params,
                        relative_dist_vec=None, factor_CRB=None):
    """
    Fisher matrix entry for 3D CRB model (Nx3 convention).

    Parameters
    ----------
    S_hov : ndarray, shape (N, 3)
        UAV hovering points
    s_target : ndarray, shape (3,)
        Target position
    entry : str
        One of ["theta_a","theta_b","theta_c",
                "theta_d","theta_e","theta_f"]
    params : dict
        System parameters
    """

    S_hov = np.asarray(S_hov, dtype=float)
    s_target = np.asarray(s_target, dtype=float)

    assert S_hov.ndim == 2 and S_hov.shape[1] == 3

    if relative_dist_vec is None:
        relative_dist_vec = relative_distance(S_hov, s_target)

    # tránh singular
    relative_dist_vec = np.maximum(relative_dist_vec, 1e-6)

    if factor_CRB is None:
        sim = params["sim"]
        factor_CRB = (sim["P"] * sim["G_p"] * sim["beta_0"]) / (
            sim["a"] * sim["sigma_0"]**2
        )

    rx = (S_hov[:, 0] - s_target[0]).reshape(-1, 1)
    ry = (S_hov[:, 1] - s_target[1]).reshape(-1, 1)
    rz = (S_hov[:, 2] - s_target[2]).reshape(-1, 1)

    D4 = np.diag(1.0 / (relative_dist_vec**4))
    D6 = np.diag(1.0 / (relative_dist_vec**6))

    if entry == "theta_a":
        return (factor_CRB * (rx.T @ D6 @ rx) + 8 * (rx.T @ D4 @ rx)).item()

    elif entry == "theta_b":
        return (factor_CRB * (ry.T @ D6 @ ry) + 8 * (ry.T @ D4 @ ry)).item()

    elif entry == "theta_c":
        return (factor_CRB * (rz.T @ D6 @ rz) + 8 * (rz.T @ D4 @ rz)).item()

    elif entry == "theta_d":   # x–y
        return (factor_CRB * (rx.T @ D6 @ ry) + 8 * (rx.T @ D4 @ ry)).item()

    elif entry == "theta_e":   # y–z
        return (factor_CRB * (ry.T @ D6 @ rz) + 8 * (ry.T @ D4 @ rz)).item()

    elif entry == "theta_f":   # x–z
        return (factor_CRB * (rx.T @ D6 @ rz) + 8 * (rx.T @ D4 @ rz)).item()

    else:
        raise ValueError("Invalid entry type")
