import numpy as np
# This function is for calculating the velocity of the UAV. Input: The initial point and the waypoints of the stage.
def calc_velocity(S, s_s, params):
    """
    Compute UAV velocity from discrete waypoints (2D or 3D).

    Parameters
    ----------
    S : np.ndarray, shape (d, N)
        UAV waypoints in d-dimensional space (d = 2 or 3)
    s_s : np.ndarray, shape (d,)
        Start point of the stage
    params : dict
        Must contain params["sim"]["T_f"]

    Returns
    -------
    V : np.ndarray, shape (d, N)
        Velocity vectors
    """

    S = np.asarray(S, dtype=float)
    s_s = np.asarray(s_s, dtype=float).reshape(-1, 1)

    # --- Shape check (VERY important for GA) ---
    if S.shape[0] != s_s.shape[0]:
        raise ValueError(
            f"Dimension mismatch: S has {S.shape[0]} dims "
            f"but s_s has {s_s.shape[0]}"
        )

    # --- Concatenate start + trajectory ---
    full_traj = np.concatenate((s_s, S), axis=1)  # (d, N+1)

    # --- Displacement ---
    displacement = np.diff(full_traj, axis=1)     # (d, N)

    # --- Time step ---
    T_f = params["sim"]["T_f"]
    if T_f <= 0:
        raise ValueError("T_f must be positive")

    # --- Velocity ---
    V = displacement / T_f
    # print(f"Kích cỡ của Vận tốc đem đi so sánh là: {V.shape}")
    return V