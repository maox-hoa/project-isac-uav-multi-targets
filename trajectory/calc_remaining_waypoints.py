import numpy as np

def calc_remaining_waypoints(
    E_remain,
    E_stg=5200,
    N_stg_ref=25,
    mu=5,
    E_reserve=750
):
    """
    Compute remaining feasible flight and hovering waypoints
    given remaining UAV energy, with safety reserve.

    Parameters
    ----------
    E_remain : float
        Remaining onboard energy
    E_stg : float
        Energy consumption of one nominal stage
    N_stg_ref : int
        Number of flight waypoints in one nominal stage
    mu : int
        Ratio N / K (flight to hover)
    E_reserve : float
        Mandatory safety energy reserve

    Returns
    -------
    N_lst : int
        Remaining feasible flight waypoints
    K_lst : int
        Remaining feasible hovering waypoints
    """

    # --- Usable energy after safety reserve ---
    E_usable = E_remain - E_reserve

    if E_usable <= 0:
        return 0, 0

    # --- Average energy per flight waypoint ---
    E_pt = E_stg / N_stg_ref

    # --- Remaining flight waypoints ---
    N_lst = int(np.floor(E_usable / E_pt))

    # --- Remaining hovering waypoints ---
    K_lst = int(np.floor(N_lst / mu))

    return N_lst, K_lst