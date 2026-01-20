import numpy as np
from comm_model.users_quad_distance import users_quad_distance

def stage_comm_data(Sj, Bm, params):
    """
    Computes total transmitted data for each CU in one stage j.

    Parameters
    ----------
    Sj : np.ndarray, shape (3, Nf_j)
        UAV trajectory in stage j
    Bm : np.ndarray, shape (M,)
        Bandwidth allocation for each CU in stage j
    params : dict
    Returns
    -------
    psi_c : np.ndarray, shape (M,)
        Total transmitted data for each CU in this stage
    """
    sim = params["sim"]
    Tf = sim["T_f"]
    sc_list = params["setup"]["comm_user_pos"]
    Th = sim["T_h"]
    mu = sim["mu"]
    P = sim["P"]
    alpha_0 = sim["alpha_0"]
    sigma_0 = sim["sigma_0"]

    M = len(sc_list)
    Nf = Sj.shape[1]
    Nh = Nf // mu

    psi_c = np.zeros(M)

    # ===== flying segments =====
    for n in range(Nf):
        for m in range(M):
            d = users_quad_distance(Sj[:, n:n+1], sc_list[m])[0]
            snr = (P * alpha_0) / (d**2 * sigma_0**2)
            psi_c[m] += Tf * Bm[m] * np.log2(1 + snr)

    # ===== hovering points =====
    for gamma in range(1, Nh + 1):
        idx = mu * gamma - 1
        for m in range(M):
            d = users_quad_distance(Sj[:, idx:idx+1], sc_list[m])[0]
            snr = (P * alpha_0) / (d**2 * sigma_0**2)
            psi_c[m] += Th * Bm[m] * np.log2(1 + snr)

    return psi_c

def comm_fairness_metric(psi_c_history):
    """
    psi_c_history: list of psi_c arrays from stage 1 to j
    """
    psi_total = np.sum(np.stack(psi_c_history, axis=0), axis=0)
    return np.min(psi_total)
