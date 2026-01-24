import numpy as np
from scipy.optimize import least_squares
import itertools

# Import hàm đo khoảng cách của bạn
from radar_sensing_model.sigma_k import sigma_k


# Giả sử hàm sense_target_3d nằm trong file tương ứng, bạn import vào nhé
# from radar_sensing_model.sense_target_3d import sense_target_3d

# =========================================================
# 1. HÀM SINH ĐIỂM HOVERING QUANH BASE
# =========================================================
def generate_stage_0_points(base_pos, radius=50, altitude=100):
    """
    Tạo 3 điểm hovering xung quanh Base Station để tam giác lượng.

    Parameters
    ----------
    base_pos : ndarray (2,) or (3,)
        Tọa độ [x, y] hoặc [x, y, z] của trạm sạc.
    radius : float
        Bán kính bay quanh base.
    altitude : float
        Độ cao bay.

    Returns
    -------
    S_init : ndarray, shape (3, 3)
        3 cột tương ứng với 3 điểm (x, y, z).
    """
    base_pos = np.array(base_pos).flatten()
    cx, cy = base_pos[0], base_pos[1]

    # Tạo 3 điểm thành hình tam giác đều quanh base
    angles = np.array([0, 120, 240]) * (np.pi / 180)

    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    z = np.full(3, altitude)

    S_init = np.vstack((x, y, z))  # Shape (3, 3)
    return S_init


# =========================================================
# 2. HÀM ĐO ĐẠC GIẢ LẬP (CÓ XÁO TRỘN ECHOES)
# =========================================================
def get_unlabeled_measurements(targets_ground_truth, S_init, params, rng=None):
    """
    Thực hiện đo tại 3 điểm hover. Tại mỗi điểm, nhận về danh sách
    các khoảng cách nhưng KHÔNG BIẾT khoảng cách nào của target nào.

    Returns
    -------
    measurements_per_point : list of lists
        Ví dụ với 3 điểm hover và 2 targets:
        [
          [d_1a, d_1b],  # Tại Hover 1 (Thứ tự đã bị xáo trộn)
          [d_2b, d_2a],  # Tại Hover 2
          [d_3a, d_3b]   # Tại Hover 3
        ]
    """
    if rng is None:
        rng = np.random.default_rng()

    N_points = S_init.shape[1]
    measurements_per_point = []

    # Hàm local mô phỏng đo đạc (dựa trên hàm bạn cung cấp)
    def simulate_sensing(s_t, s_q, p):
        diff = s_t.reshape(3, 1) - s_q.reshape(3, 1)
        dist = np.linalg.norm(diff)
        sig = sigma_k(dist, p)
        noise = rng.standard_normal()
        return dist + sig * noise

    for i in range(N_points):
        S_curr = S_init[:, i]  # Điểm hover hiện tại
        dists_at_node = []

        # Đo tất cả các target
        for t_pos in targets_ground_truth:
            d = simulate_sensing(t_pos, S_curr, params)
            dists_at_node.append(d)

        # QUAN TRỌNG: Xáo trộn thứ tự echoes để giả lập thực tế
        # Ta nhận được 1 tập tín hiệu nhưng không có nhãn
        rng.shuffle(dists_at_node)
        measurements_per_point.append(np.array(dists_at_node))

    return measurements_per_point


# =========================================================
# 3. GIẢI THUẬT GÁN VÀ ƯỚC LƯỢNG (DATA ASSOCIATION & ESTIMATION)
# =========================================================
def solve_position_from_distances(S_points, distances, z_target=0):
    """
    Hàm tối ưu tìm vị trí (x,y) thỏa mãn bộ 3 khoảng cách.
    Sử dụng Grid Search để tìm Initial Guess tốt nhất, sau đó dùng Least Squares.
    """
    # 1. Flatten inputs
    distances = np.array(distances).flatten()
    S_points = np.array(S_points)

    # Hàm tính residuals (giữ nguyên logic)
    def residuals(vars):
        x, y = vars
        est_pos = np.array([x, y, z_target])
        d_est = np.linalg.norm(S_points - est_pos.reshape(3, 1), axis=0)
        return d_est - distances

    # 2. [CẢI TIẾN] Grid Search để tìm điểm khởi tạo tốt nhất (Global Initialization)
    # Thay vì đoán mò ở Base, ta rải lưới 4x4 hoặc 5x5 trên bản đồ 1500x1500m
    L_x, L_y = 1500, 1500  # Kích thước bản đồ (theo params của bạn)
    grid_x = np.linspace(0, L_x, 6)  # 0, 300, 600, 900, 1200, 1500
    grid_y = np.linspace(0, L_y, 6)

    best_x0 = None
    best_initial_cost = float('inf')

    # Duyệt qua lưới để tìm vùng khả dĩ nhất
    for gx in grid_x:
        for gy in grid_y:
            res = residuals([gx, gy])
            cost = 0.5 * np.sum(res ** 2)  # Hàm chi phí Least Squares
            if cost < best_initial_cost:
                best_initial_cost = cost
                best_x0 = np.array([gx, gy])

    # 3. Chạy Least Squares từ điểm khởi tạo tốt nhất tìm được
    # Thêm bounds để đảm bảo không văng ra khỏi bản đồ
    result = least_squares(
        residuals,
        best_x0,
        bounds=([0, 0], [L_x, L_y]),  # Giới hạn tìm kiếm trong bản đồ
        ftol=1e-4, xtol=1e-4  # Tăng độ nhạy hội tụ
    )

    # Trả về kết quả
    estimated_xy = result.x
    cost = result.cost
    return np.array([estimated_xy[0], estimated_xy[1], z_target]), cost
def perform_initial_estimation(S_init, measurements_list, num_targets):
    """
    Core Logic: Thử mọi tổ hợp ghép cặp để tìm ra các vị trí có lý nhất.

    Logic:
    1. Tại Hover 1 có K đo đạc. Hover 2 có K. Hover 3 có K.
    2. Tổng tổ hợp có thể xảy ra là K * K * K.
    3. Thử từng tổ hợp, giải tìm tọa độ. Nếu Cost nhỏ -> Tổ hợp hợp lệ (Valid Candidate).
    """

    # Tạo tất cả các chỉ số tổ hợp. Ví dụ 2 target -> indices [0, 1]
    # Cartesian product của [0,1] x [0,1] x [0,1] => (0,0,0), (0,0,1), ...
    idxs = range(num_targets)
    combinations = list(itertools.product(idxs, repeat=3))  # 3 là số điểm hover

    candidates = []

    for comb in combinations:
        # comb là (idx_at_h1, idx_at_h2, idx_at_h3)
        # Lấy ra bộ 3 khoảng cách tương ứng với tổ hợp này
        d_set = [
            measurements_list[0][comb[0]],
            measurements_list[1][comb[1]],
            measurements_list[2][comb[2]]
        ]

        # Giải tìm vị trí
        pos_est, cost = solve_position_from_distances(S_init, d_set)

        candidates.append({
            "comb": comb,
            "pos": pos_est,
            "cost": cost
        })

    # Sắp xếp các ứng viên theo sai số (cost) tăng dần
    candidates.sort(key=lambda x: x["cost"])

    # Chọn ra num_targets ứng viên tốt nhất KHÔNG trùng lặp phép đo
    # (Đây là thuật toán Greedy đơn giản cho bài toán Assignment)
    final_estimates = []
    used_indices_h1 = set()

    print(f"--- Debug Association Stage 0 ---")
    for cand in candidates:
        comb = cand["comb"]
        # Kiểm tra xem phép đo tại H1 trong tổ hợp này đã được dùng cho target khác chưa?
        # (Để chặt chẽ hơn thì check cả H2, H3, nhưng check H1 thường đủ phân loại track)
        if comb[0] not in used_indices_h1:
            if len(final_estimates) < num_targets:
                final_estimates.append(cand["pos"])
                used_indices_h1.add(comb[0])
                print(f"Matched Combinations {comb} | Error: {cand['cost']:.4f} | Est: {cand['pos']}")

    return np.array(final_estimates)


# =========================================================
# 4. HÀM CHẠY CHÍNH (WRAPPER)
# =========================================================
def run_stage_0(params, true_targets):
    """
    Chạy toàn bộ quy trình Stage 0.
    """
    print("=== BẮT ĐẦU GIAI ĐOẠN 0 (INITIALIZATION) ===")

    # 1. Setup
    base_pos = params["setup"]["base_station_pos"]
    num_targets = len(true_targets)

    # 2. Sinh 3 điểm Hover
    S_init = generate_stage_0_points(base_pos, radius=100, altitude=params["sim"]["H"])
    print(f"Generated 3 Initial Hover Points around Base {base_pos}")

    # 3. Đo đạc (Kết quả bị xáo trộn)
    measurements_blind = get_unlabeled_measurements(true_targets, S_init, params)
    print("Received echoes (blind/shuffled) from 3 points.")

    # 4. Giải bài toán gán và ước lượng
    estimated_positions = perform_initial_estimation(S_init, measurements_blind, num_targets)

    print("\n=== KẾT QUẢ GIAI ĐOẠN 0 ===")
    for i, est in enumerate(estimated_positions):
        print(f"Target {i + 1} Est: {est} (Ground Z=0 assumed)")

    return estimated_positions