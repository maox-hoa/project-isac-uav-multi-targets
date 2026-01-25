import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import permutations

# Giữ nguyên các import của bạn
from radar_sensing_model.sense_target_3d import sense_target_3d
from mle.estimate_target_3d import estimate_target_3d
from trajectory.calc_real_energy_3d import calc_real_energy_3d
from trajectory.calc_remaining_waypoints import calc_remaining_waypoints
from main_algorithms.ga_algorithm import optimize_ga


def multi_stage_3d(params, setup):
    """
    Multi-stage UAV trajectory optimization in 3D.
    Includes a dedicated LAST-STAGE energy-matching optimization.
    ADDED: Stage 0 (Coarse localization) logic.
    """

    # =========================
    # Simulation parameters
    # =========================
    mu = params["sim"]["mu"]
    N_stg = params["sim"]["N_stg"]
    K_stg = params["sim"]["K_stg"]

    # Lấy số lượng target K từ setup hoặc params (đoán theo cấu trúc)
    # Giả sử s_t là mảng (3, K)
    K = setup["sense_target_pos"].shape[1]

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
    D_meas_list = []  # mỗi phần tử: (K,)
    S_opt_list = []  # mỗi phần tử: (3, N_i)
    S_target_est_list = []  # mỗi phần tử: (3,)

    E_m_vec = []
    E_used_vec = []

    fitness_history_all = []
    rate_history_all = []
    crb_history_all = []

    # =========================
    # STAGE 0: COARSE LOCALIZATION
    # =========================
    print("=== BẮT ĐẦU GIAI ĐOẠN 0 (COARSE LOCALIZATION) ===")

    # 1. Helper: Hàm tính năng lượng thủ công cho Stage 0 (vì Tf thay đổi)
    def calc_stage0_energy(S_waypoints, v_max, t_h):
        """
        Tính năng lượng cho Stage 0 bay với vận tốc v_max.
        Giả sử mô hình năng lượng đơn giản: P_hover cố định, P_fly thay đổi theo v.
        Bạn có thể thay thế phần này bằng hàm calc_real_energy_3d nếu nó hỗ trợ tốc độ biến thiên.
        """
        # Tham số năng lượng giả định (cần chỉnh khớp với params của bạn)
        P_hover = 88.6  # Watts (thường là PI)
        P_parasitic = 80.0  # Watts (thường là P0 khi hover + phần ma sát)

        total_energy = 0.0
        for i in range(1, S_waypoints.shape[1]):
            # Tính khoảng cách và thời gian bay
            dist = np.linalg.norm(S_waypoints[:, i] - S_waypoints[:, i - 1])
            t_flight = dist / v_max

            # Năng lượng bay (Giả định P_fly ~ P_hover + P_air_drag)
            # Với v_max, năng lượng tiêu thụ cao hơn hover.
            # Mô hình đơn giản hóa: E = Power * time
            # Ở đây dùng P_parasitic công suất động cơ bay
            E_fly = (P_hover + P_parasitic) * t_flight

            # Tính năng lượng hover tại điểm đích (trừ điểm cuối cùng vì mission kết thúc hoặc chuyển stage)
            if i < S_waypoints.shape[1]:
                E_hover = P_hover * t_h
            else:
                E_hover = 0

            total_energy += (E_fly + E_hover)

        return total_energy

    # 2. Helper: Hàm Ước lượng thô (MLE + Association) cho Stage 0
    def estimate_coarse_targets(S_hover, measurements_list, params):
        # measurements_list: list chứa K mảng, mỗi mảng là khoảng cách đo được tại 1 Hover Point
        # Đầu vào: S_hover shape (3, 3) - 3 điểm hover
        # Dữ liệu: measurements_list[0] là mảng K distances tại HP1...

        K_t = len(measurements_list[0])
        best_estimates = []
        min_total_nll = float('inf')

        # Duyệt tất cả hoán vị để gán echo cho target
        perms = list(permutations(range(K_t)))

        # Điểm khởi đầu cho optimize (lấy trung tâm các hover points + z=0)
        x0_init = np.mean(S_hover, axis=1)
        x0_init[2] = 0  # Ground

        bounds = [
            (0.0, params["sim"]["L_x"]),
            (0.0, params["sim"]["L_y"]),
            (0.0, 0.0),  # Target ở mặt đất
        ]

        for p in perms:
            current_estimates = []
            current_nll = 0
            valid = True

            # Với mỗi target k
            for k in range(K_t):
                # Lấy bộ 3 khoảng cách tương ứng từ 3 Hover Points
                # measurements_list[0][k]: dist từ HP1
                # measurements_list[1][p[k]]: dist từ HP2 (đã gán theo hoán vị)
                # measurements_list[2][p[k]]: dist từ HP3 (đã gán theo hoán vị)

                # Lưu ý: Ở đây ta giả sử thứ tự list[0] là chuẩn, ta hoán vị list[1], list[2]
                # Code này giả định measurements_list được truyền vào đúng thứ tự [HP1, HP2, HP3]

                # Tuy nhiên để chính xác theo thuật toán Brute-force trước đó:
                # Cần thử ghép bộ (HP1[i], HP2[j], HP3[k]) thành 1 target.
                # Ở đây đơn giản hóa: Giả sử thứ tự HP1 là chuẩn, thử hoán vị HP2 và HP3.

                d_set = [
                    measurements_list[0][k],
                    measurements_list[1][p[k]],
                    measurements_list[2][p[k]]
                ]

                def nll(theta):
                    d_pred = np.sqrt(
                        (S_hover[0, :] - theta[0]) ** 2 +
                        (S_hover[1, :] - theta[1]) ** 2 +
                        (S_hover[2, :] - theta[2]) ** 2
                    )
                    return np.sum((np.array(d_set) - d_pred) ** 2)

                res = minimize(nll, x0_init, method="L-BFGS-B", bounds=bounds)

                if not res.success:
                    valid = False
                    break

                current_estimates.append(res.x)
                current_nll += res.fun

            if valid and current_nll < min_total_nll:
                min_total_nll = current_nll
                best_estimates = current_estimates

        return np.array(best_estimates).T

    # 3. Tạo Stage 0 Trajectory
    # Thiết lập
    R0 = 300.0  # Bán kính 300m
    # Lấy độ cao bay H từ params hoặc setup (mặc định 200m nếu không có)
    H_stage0 = params["sim"].get("H", 200.0)
    V_max = params["sim"].get("V_max", 20.0)  # Vận tốc tối đa m/s
    T_h = params["sim"].get("T_h", 1.0)

    # Tính toán 3 điểm tam giác đều xung quanh Base (s_b)
    # Base có thể ở z=0 hoặc z=H, ta nâng UAV lên z=H ngay khi bắt đầu
    angles = np.deg2rad([0, 120, 240])
    waypoints_stage0 = [s_b]  # Điểm bắt đầu

    for ang in angles:
        pt = [
            s_b[0] + R0 * np.cos(ang),
            s_b[1] + R0 * np.sin(ang),
            H_stage0
        ]
        waypoints_stage0.append(pt)

    S_stage0 = np.array(waypoints_stage0).T  # Shape (3, 4) [Base, HP1, HP2, HP3]

    # 4. Sensing tại 3 Hover Points
    # Loại bỏ điểm Base (index 0) vì chỉ cần 3 điểm lơ lửng
    S_hover_stage0 = S_stage0[:, 1:]
    meas_stage0 = []

    print(f"Stage 0: Sensing tại {S_hover_stage0.shape[1]} điểm...")
    for i in range(S_hover_stage0.shape[1]):
        pos_uav = S_hover_stage0[:, i].reshape(3, 1)
        # sense_target_3d trả về mảng khoảng cách shape (K,)
        d = sense_target_3d(s_t, pos_uav, params)
        meas_stage0.append(d)

    # 5. Estimate (MLE + Association)
    print("Stage 0: Thực hiện Association và Ước lượng vị trí thô...")
    est_pos = estimate_coarse_targets(S_hover_stage0, meas_stage0, params)

    # Cập nhật vị trí ước lượng cho các vòng lặp sau
    s_target_est = est_pos
    print(f"Stage 0: Vị trí ước lượng ban đầu (Coarse Est):\n{s_target_est.T}")

    # 6. Cập nhật Năng lượng
    print("Stage 0: Tính toán năng lượng tiêu hao...")
    E_used_stage0 = calc_stage0_energy(S_stage0, V_max, T_h)

    E_m -= E_used_stage0
    E_used_vec.append(E_used_stage0)
    E_m_vec.append(E_m)

    # Cập nhật vị trí hiện tại (s_s) về điểm cuối cùng của Stage 0
    s_s = S_stage0[:, -1]

    # Lưu kết quả Stage 0 (Nếu muốn lưu vào list chung, lưu ý logic mu trong vòng lặp chính)
    # Ở đây ta lưu riêng để tránh lỗi index khi vòng lặp chính chạy optimize_ga với mu
    S_opt_list.append(S_stage0)

    # Lưu trữ đo lường Stage 0 vào D_meas_list để MLE sau này có thể dùng (nếu cần)
    # Tuy nhiên, do Stage 0 không tuân thủ mu, việc ghép vào list chính
    # có thể gây lỗi trong hàm estimate_target_3d (vòng lặp chính) nếu nó phụ thuộc vào cấu trúc N_stg.
    # Cách an toàn: Ta chỉ dùng Stage 0 để khởi tạo s_target_est.
    # Các đo lường trong các Stage 1, 2... sẽ dựa trên s_target_est này.

    print(f"Stage 0 hoàn tất. Năng lượng còn lại: {E_m / 1000:.2f} kJ")
    print("=========================================================\n")

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
        # Lưu ý: optimize_ga sẽ nhận điểm bắt đầu là s_s (điểm cuối Stage 0)
        wp_new, fitness_crb_history, fitness_rate_history = optimize_ga_3d(
            S_total_m, s_target_est, params, E_m, N_wp=N_cur
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
        D_stage = sense_target_3d(s_t, hover_positions, params)
        D_meas_list.append(D_stage)

        S_hover_all = np.concatenate(
            [S[:, mu - 1::mu] for S in S_opt_list], axis=1
        )
        D_all = np.concatenate(D_meas_list)

        s_target_est = estimate_target_3d(
            S_hover_all, D_all, params, s_target_est
        )
        S_target_est_list.append(s_target_est)

        # --------------------------------------------------
        # Energy update
        # --------------------------------------------------
        E_used = calc_real_energy_3d(wp_new, s_s, params)

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
        "average_reachable_rate": max(rate_history_all) if len(rate_history_all) > 0 else 0,
        "average_reachable_crb": max(crb_history_all) if len(crb_history_all) > 0 else 0,
    }