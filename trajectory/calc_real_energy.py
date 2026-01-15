import numpy as np
from trajectory.calc_velocity import calc_velocity


# ==========================================================
# Power model (UNCHANGED – dimension independent)
# ==========================================================
def power_model(V_norm, P_0, U_tip, P_I, v_0, D_0, rho, s, A):
    """
    Numerically safe propulsion power model.
    Depends only on speed magnitude ||v||.
    """

    # --- Tip-speed physical limit ---
    V_norm = np.clip(V_norm, 0.0, 0.99 * U_tip)

    # --- Profile power ---
    term1 = P_0 * (1.0 + 3.0 * (V_norm ** 2) / (U_tip ** 2))

    # --- Induced power ---
    with np.errstate(over="ignore", invalid="ignore"):
        inner = 1.0 + (V_norm ** 4) / (4.0 * (v_0 ** 4))
        inner = np.maximum(inner, 1.0)
        inner_sqrt = np.sqrt(inner) - (V_norm ** 2) / (2.0 * (v_0 ** 2))
        inner_sqrt = np.maximum(inner_sqrt, 0.0)

    term2 = P_I * np.sqrt(inner_sqrt)

    # --- Parasitic power ---
    term3 = 0.5 * D_0 * rho * s * A * (V_norm ** 2)

    P_total = term1 + term2 + term3
    P_total = np.where(np.isfinite(P_total), P_total, 1e12)

    return P_total


# ==========================================================
# Real energy computation – 3D UAV motion
# ==========================================================
def calc_real_energy_3d(S, s_s, params):
    """
    Compute total energy consumption of one stage (3D UAV trajectory).

    Parameters
    ----------
    S : ndarray, shape (3, N)
        UAV waypoints in 3D
    s_s : ndarray, shape (3,)
        Start position of the stage
    params : dict

    Returns
    -------
    E_used : float
        Total energy consumption
    """

    # === Energy parameters ===
    e = params["energy"]
    s = e["s"]
    A = e["A"]
    rho = e["rho"]
    D_0 = e["D_0"]
    v_0 = e["v_0"]
    U_tip = e["U_tip"]
    P_0 = e["P_0"]
    P_I = e["P_I"]

    # === Simulation parameters ===
    sim = params["sim"]
    T_f = sim["T_f"]
    T_h = sim["T_h"]
    K_stg = sim["K_stg"]

    # === Velocity (3D) ===
    V = calc_velocity(S, s_s, params)   # shape (3, N)
    V_norm = np.linalg.norm(V, axis=0)

    V_norm = np.clip(V_norm, 0.0, 0.99 * U_tip)

    # === Power ===
    flight_power = power_model(
        V_norm, P_0, U_tip, P_I, v_0, D_0, rho, s, A
    )

    hover_power = power_model(
        np.zeros(K_stg),
        P_0, U_tip, P_I, v_0, D_0, rho, s, A
    )

    # === Energy ===
    E_used = T_f * np.sum(flight_power) + T_h * np.sum(hover_power)

    if not np.isfinite(E_used):
        return 1e12

    return E_used