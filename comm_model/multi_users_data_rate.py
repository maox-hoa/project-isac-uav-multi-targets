import numpy as np
from comm_model.users_quad_distance import users_quad_distance


def stage_comm_data(Sj, Bm, params):
    """
    Computes total transmitted data for each CU in one stage j.

    Sj : np.ndarray, shape (3, Nf)
        UAV trajectory in stage j
    Bm : np.ndarray, shape (M,)
        Bandwidth allocation for each CU
    params : dict
    """

    sim = params["sim"]

    Tf = sim["T_f"]
    Th = sim["T_h"]
    mu = sim["mu"]

    P = sim["P"]
    alpha_0 = sim["alpha_0"]
    sigma_0 = sim["sigma_0"]      # shape (M,)

    sc_list = params["setup"]["comm_user_pos"]  # shape (M, 3)

    M = len(sc_list)
    Nf = Sj.shape[1]
    Nh = Nf // mu

    psi_c = np.zeros(M)

    # =========================
    # Flying segments
    # =========================
    for m in range(M):
        d = users_quad_distance(Sj, sc_list[m])   # shape (Nf,)

        sigma_m = sigma_0[m]                     # scalar for user m

        snr = (P * alpha_0) / (d**2 * sigma_m**2)

        psi_c[m] += Tf * Bm[m] * np.sum(np.log2(1 + snr))

    # =========================
    # Hovering points
    # =========================
    for gamma in range(1, Nh + 1):
        idx = mu * gamma - 1

        for m in range(M):
            d = users_quad_distance(Sj[:, idx:idx+1], sc_list[m])[0]

            sigma_m = sigma_0[m]

            snr = (P * alpha_0) / (d**2 * sigma_m**2)

            psi_c[m] += Th * Bm[m] * np.log2(1 + snr)

    return psi_c
