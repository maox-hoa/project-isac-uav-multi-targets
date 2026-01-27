# Đã test thành công.
import numpy as np

def g_k(d_s, beta_0):
    """
    Two-way channel power gain.

    parameters
    ----------
    d_s : float or np.ndarray
        Distance(s) to sensing target.
    beta_0 : float
        Reference channel power at distance d_s = 1 meter.

    Returns
    -------
    y : np.ndarray or float
        Channel power gain.
    """
    d_s = np.asarray(d_s, dtype=float)
    return beta_0 / (d_s ** 4)

def sigma_k(d_s, params):
    """
    python version of MATLAB sigma_k.m

    parameters
    ----------
    d_s : array_like
        Distances to sensing target.
    params : dict
        Must contain:
            params["sim"]["p"]
            params["sim"]["G_p"]
            params["sim"]["beta_0"]
            params["sim"]["a"]
            params["sim"]["sigma_0"]

    Returns
    -------
    sig_k : np.ndarray
        Standard deviation vector (same size as d_s)
    """
    d_s = np.asarray(d_s, dtype=float)

    sim = params["sim"]
    p = sim["P"]
    g_p= sim["G_p"]
    beta_0 = sim["beta_0"]
    a = sim["a"]
    sigma_0 = sim["sigma_0"]

    # g_k is already vectorized
    g_val = g_k(d_s, beta_0)

    sig_k = np.sqrt((a * (sigma_0 ** 2)) / (p * g_p* g_val))
    return sig_k