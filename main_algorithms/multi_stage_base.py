import numpy as np

from mle.multi_targets_sensing import sense_two_targets
from mle.multi_targets_sensing import associate_measurements
from trajectory.calc_real_energy import calc_real_energy
from trajectory.calc_remaining_waypoints import calc_remaining_waypoints
from main_algorithms.ga_algorithm import optimize_ga
from mle.estimate_target import estimate_target
def multi_stage_3d(params, setup):
    """
    Multi-stage UAV trajectory optimization in 3D.
    Includes a dedicated LAST-STAGE energy-matching optimization.
    """

    # =========================
    # Simulation parameters
    # =========================
    mu     = params["sim"]["mu"]
    N_stg  = params["sim"]["N_stg"]
    K_stg  = params["sim"]["K_stg"]

    M_max = 20

    # Energy thresholds
    E_min = 7e3

    # =========================
    # Setup (3D)
    # =========================
    s_b = setup["base_station_pos"].copy()
    s_s = s_b.copy()

    s_t = setup["sense_target_pos"]
    s_target_est = setup["est_sense_target"].copy()

    E_m = setup["total_energy"]

    # =========================
    # Storage
    # =========================
    # D_meas_list = np.full((K_stg, M_max), np.nan)
    # S_opt_list = np.full((3, N_stg, M_max), np.nan)
    # S_target_est_list = np.full((3, M_max), np.nan)
    D_meas_list = []  # mỗi phần tử: (K_i,)
    S_opt_list = []  # mỗi phần tử: (3, N_i)
    S_target_est_list = []  # mỗi phần tử: (3,)

    E_m_vec = []
    E_used_vec = []

    fitness_history_all = []
    rate_history_all = []
    crb_history_all = []

    # =========================
    # Main loop
    # =========================
    S_total_m = None
    m = 0
    while True:
        m += 1
        print(f"Bắt đầu giai đoạn {m}")

        # --------------------------------------------------
        # Build past trajectory
        # --------------------------------------------------
        if len(S_opt_list) == 0:
            S_total_m = None
        else:
            S_total_m = np.concatenate(S_opt_list, axis=1)

        # --------------------------------------------------
        # Check for last stage
        # --------------------------------------------------
        is_last_stage = False
        N_cur = N_stg

        if E_m <= E_min:
            N_cur, K_cur = calc_remaining_waypoints(E_m)
            is_last_stage = True

            if N_cur <= 0:
                print("Không đủ năng lượng cho stage cuối – dừng.")
                break

            print(f"Final stage | N_lst={N_cur}, K_lst={K_cur}")

        # --------------------------------------------------
        # Optimize trajectory
        # --------------------------------------------------
        wp_new, fitness_crb_history, fitness_rate_history = optimize_ga(
            S_total_m, s_target_est, params, E_m, N_wp = N_cur
        )

        rate_history_all.extend(fitness_rate_history)
        crb_history_all.extend(fitness_crb_history)

        # --------------------------------------------------
        # Save trajectory
        # --------------------------------------------------
        S_opt_list.append(wp_new)

        # --------------------------------------------------
        # Sense & estimate
        # --------------------------------------------------
        hover_positions = wp_new[:, mu - 1::mu]
        D_stage = sense_two_targets(s_t, hover_positions, params)
        D_meas_list.append(D_stage)

        S_hover_all = np.concatenate(
            [S[:, mu - 1::mu] for S in S_opt_list], axis=1
        )
        D_all = np.concatenate(D_meas_list)

        s_target_est = estimate_target(
            S_hover_all, D_all, params, s_target_est
        )
        S_target_est_list.append(s_target_est)

        # --------------------------------------------------
        # Energy update
        # --------------------------------------------------
        E_used = calc_real_energy(wp_new, s_s, params)

        E_used_vec.append(E_used)
        E_m -= E_used
        E_m_vec.append(E_m)

        s_s = wp_new[:, -1]

        if is_last_stage:
            print("Hoàn thành stage cuối – kết thúc mission.")
            break
    # =========================
    # Final bookkeeping
    # =========================
    M_final = m
    N_tot = sum(S.shape[1] for S in S_opt_list)
    K_tot = sum(S.shape[1] // mu for S in S_opt_list)
    return {
        "E_used_vec": E_used_vec,
        "E_m_vec": E_m_vec,
        "S_opt_list": S_opt_list,
        "S_target_est_list": S_target_est_list,
        "D": D_meas_list,
        "M": M_final,
        "N_tot": N_tot,
        "K_tot": K_tot,
        "rate_history": np.array(rate_history_all),
        "crb_history": np.array(crb_history_all),
        "average_reachable_rate": max(rate_history_all) ,
        "average_reachable_crb": max(crb_history_all) ,
    }