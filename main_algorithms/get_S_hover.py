import numpy as np

def get_S_hover(params, m_start, m_end, S):
    """
    Select hovering points from a UAV trajectory (2D or 3D).

    Parameters
    ----------
    params : dict
        Must contain:
            params["sim"]["N_stg"]
            params["sim"]["mu"]
    m_start : int
        Start stage index (1-based)
    m_end : int
        End stage index (1-based)
    S : np.ndarray, shape (d, N)
        UAV trajectory, d = 2 or 3

    Returns
    -------
    hover_idxs : np.ndarray
        Indices of hovering points (1D array)
    S_hover : np.ndarray
        Hovering points, shape (d, K)
    """

    N_stg = params["sim"]["N_stg"]
    mu    = params["sim"]["mu"]
    K_stg = N_stg // mu

    # --- Shift using modulo ---
    shft = N_stg % mu

    indices = np.arange(m_start - 1 + mu, m_end - 1 + mu + 1)
    shft_vec = shft * np.mod(indices, mu)

    shft_mat = np.tile(shft_vec, (K_stg, 1))
    idxs_shift = shft_mat.reshape(-1)

    # --- Linear hovering indices ---
    hover_idxs_linear = np.arange(
        mu + (m_start - 1) * K_stg * mu,
        (m_end * K_stg * mu) + 1,
        mu
    )

    hover_idxs = idxs_shift + hover_idxs_linear
    hover_idxs = hover_idxs.astype(int)

    # --- Extract hover points ---
    if S is not None and S.size != 0:
        S = np.asarray(S)
        if S.ndim != 2:
            raise ValueError("S must be a 2D array of shape (d, N)")
        S_hover = S[:, hover_idxs]
    else:
        S_hover = np.array([[]])

    return hover_idxs, S_hover