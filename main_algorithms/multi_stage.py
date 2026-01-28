import numpy as np

from mle.multi_targets_sensing import sense_two_targets
from mle.multi_targets_sensing import associate_measurements
from trajectory.calc_real_energy import calc_real_energy
from trajectory.calc_remaining_waypoints import calc_remaining_waypoints
from main_algorithms.ga_algorithm import optimize_ga
from mle.estimate_target import estimate_target
def multi_stage_3d_multi_target_multi_user(params, setup):
    """
    Multi-stage UAV trajectory optimization
    - 3D
    - 2 sensing targets
    - multiple communication users
    - energy-aware last stage
    """

    # =========================
    # Simulation parameters
    # =========================
    mu    = params["sim"]["mu"]
    N_stg = params["sim"]["N_stg"]

    M_max = 20
    E_min = 7e3

    # =========================
    # Setup
    # =========================
    s_bs = setup["base_station_pos"].copy()
    s_uav = s_bs.copy()

    s_targets_true = setup["true_target_pos"]      # (3, 2)
    s_targets_est  = setup["est_target_pos"].copy()# (3, 2)
    s_users        = setup["user_pos"]              # (3, U)

    E_m = setup["total_energy"]

    # =========================
    # Storage
    # =========================
    S_opt_list = []
    D_meas_list = []          # list of (2*K_i,)
    S_target_est_list = []    # list of (3,2)

    E_m_vec = []
    E_used_vec = []

    rate_history_all = []
    crb_history_all = []

    # =========================
    # Main loop
    # =========================
    m = 0
    while True:
        m += 1
        print(f"Bắt đầu giai đoạn {m}")

        # --------------------------------------------------
        # Past trajectory
        # --------------------------------------------------
        S_total = None if len(S_opt_list) == 0 \
            else np.concatenate(S_opt_list, axis=1)

        # --------------------------------------------------
        # Last stage check
        # --------------------------------------------------
        is_last_stage = False
        N_cur = N_stg

        if E_m <= E_min:
            N_cur, _ = calc_remaining_waypoints(E_m)
            is_last_stage = True

            if N_cur <= 0:
                print("Không đủ năng lượng – dừng mission.")
                break

            print(f"Final stage | N_lst={N_cur}")

        # --------------------------------------------------
        # GA optimization (multi-target + multi-user)
        # --------------------------------------------------
        wp_new, crb_hist, rate_hist = optimize_ga(
            S_total=S_total,
            s_targets_est=s_targets_est,
            s_users=s_users,
            params=params,
            E_remain=E_m,
            N_wp=N_cur
        )

        crb_history_all.extend(crb_hist)
        rate_history_all.extend(rate_hist)

        S_opt_list.append(wp_new)

        # --------------------------------------------------
        # Sensing
        # --------------------------------------------------
        hover_pos = wp_new[:, mu - 1::mu]

        # returns stacked measurements: (2*K,)
        D_stage = sense_two_targets(
            s_targets_true, hover_pos, params
        )
        D_meas_list.append(D_stage)

        # --------------------------------------------------
        # Association + Estimation
        # --------------------------------------------------
        S_hover_all = np.concatenate(
            [S[:, mu - 1::mu] for S in S_opt_list], axis=1
        )
        D_all = np.concatenate(D_meas_list)

        # association: reorder measurements per target
        D_assoc = associate_measurements(
            S_hover_all, D_all, params
        )

        s_targets_est = estimate_target(
            S_hover_all,
            D_assoc,
            params,
            s_targets_est
        )
        S_target_est_list.append(s_targets_est.copy())

        # --------------------------------------------------
        # Energy update
        # --------------------------------------------------
        E_used = calc_real_energy(wp_new, s_uav, params)
        E_used_vec.append(E_used)

        E_m -= E_used
        E_m_vec.append(E_m)

        s_uav = wp_new[:, -1]

        if is_last_stage:
            print("Hoàn thành stage cuối.")
            break

    # =========================
    # Final bookkeeping
    # =========================
    N_tot = sum(S.shape[1] for S in S_opt_list)
    K_tot = sum(S.shape[1] // mu for S in S_opt_list)

    return {
        "S_opt_list": S_opt_list,
        "S_target_est_list": S_target_est_list,
        "D": D_meas_list,
        "E_used_vec": E_used_vec,
        "E_m_vec": E_m_vec,
        "N_tot": N_tot,
        "K_tot": K_tot,
        "M": m,
        "rate_history": np.array(rate_history_all),
        "crb_history": np.array(crb_history_all),
        "best_rate": np.max(rate_history_all),
        "best_crb": np.min(crb_history_all),
    }
