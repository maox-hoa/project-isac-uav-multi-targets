import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import permutations
import random

# --- 1. Cấu hình tham số mô phỏng ---
np.random.seed()  # Để kết quả có thể tái hiện
random.seed()

params = {
    "sim": {
        "L_x": 1500.0,  # Chiều dài khu vực (m)
        "L_y": 1500.0,  # Chiều rộng khu vực (m)
        "L_z": 500.0,  # Độ cao tối đa (m)
        "sigma_0": 10.0,  # Độ lệch chuẩn nhiễu đo khoảng cách (m)
    },
    "uav": {
        "altitude": 200.0,  # Độ cao bay H
        "radius": 300.0  # Bán kính tam giác quanh base (m)
    }
}

# Vị trí Base Charging Station (ở giữa khu vực)
base_pos = np.array([750.0, 750.0])

# Số lượng Target (Vật thể) cần định vị
K_targets = 2

# Tạo vị trí NGẫu nhiên cho các Target (chỉ dùng để kiểm tra kết quả, không đưa cho thuật toán)
# Giả sử target nằm trên mặt đất (z=0)
true_targets = np.zeros((3, K_targets))
for k in range(K_targets):
    # Random vị trí trong bán kính 400m quanh base để đảm bảo dễ đo
    r = np.random.uniform(50, 400)
    theta = np.random.uniform(0, 2 * np.pi)
    true_targets[0, k] = base_pos[0] + r * np.cos(theta)
    true_targets[1, k] = base_pos[1] + r * np.sin(theta)
    true_targets[2, k] = 0.0  # Mặt đất

print(f"Vị trí thật của Targets:\n{true_targets.T}")


# --- 2. Hàm tạo Hovering Points (Stage 0) ---
def get_stage0_hover_points(base_xy, altitude, radius):
    """
    Tạo 3 điểm lơ lửng tạo thành tam giác đều quanh base.
    """
    points = []
    # 0 độ, 120 độ, 240 độ
    angles = np.deg2rad([0, 120, 240])
    for ang in angles:
        px = base_xy[0] + radius * np.cos(ang)
        py = base_xy[1] + radius * np.sin(ang)
        points.append([px, py, altitude])
    return np.array(points).T  # Shape (3, 3) -> rows: x, y, z


S_hover = get_stage0_hover_points(base_pos, params["uav"]["altitude"], params["uav"]["radius"])
print(f"\nTọa độ Hover Points (3 điểm):\n{S_hover.T}")


# --- 3. Hàm Cảm biến (Sensing) ---
# (Đã viết lại để chạy độc lập không cần import file ngoài)
def sigma_k(d_s, params):
    # Mô hình nhiễu đơn giản: sigma tăng nhẹ theo khoảng cách (tùy chọn)
    # Ở đây dùng sigma_0 cố định cho đơn giản như code gốc
    return np.full_like(d_s, params["sim"]["sigma_0"])

def sense_multiple_targets(S_q, true_targets_pos, params, rng=None):
    """
    Mô phỏng đo khoảng cách từ 1 vị trí UAV đến Nhiều Target.
    Trả về danh sách khoảng cách đã bị trộn thứ tự (shuffled).
    """
    if rng is None:
        rng = np.random.default_rng()

    K = true_targets_pos.shape[1]
    measurements = []

    # Lặp qua từng target để tính khoảng cách thật
    for k in range(K):
        s_t = true_targets_pos[:, k].reshape(3, 1)
        diff = s_t - S_q
        # Khoảng cách Euclidean
        d_true = np.linalg.norm(diff, axis=0)

        # Thêm nhiễu
        sig = sigma_k(d_true, params)
        noise = rng.standard_normal(size=d_true.shape)
        d_meas = d_true + sig * noise

        # Chỉ lấy giá trị vô hướng (do S_q chỉ là 1 điểm tại thời điểm tính)
        measurements.append(d_meas[0])

    # Trộn thứ tự các echo (giả lập hệ thống chưa biết echo nào của ai)
    rng.shuffle(measurements)
    return np.array(measurements)


# --- 4. Thu thập dữ liệu tại 3 điểm Hover ---
all_measurements_per_hover = []
rng = np.random.default_rng(42)

for i in range(3):
    pos_uav = S_hover[:, i].reshape(3, 1)  # Vị trí UAV hiện tại
    meas = sense_multiple_targets(pos_uav, true_targets, params, rng)
    all_measurements_per_hover.append(meas)

print(f"\nDữ liệu đo được (đã trộn) tại 3 điểm:")
for i, m in enumerate(all_measurements_per_hover):
    print(f"HP {i + 1}: {m}")


# --- 5. Echo Association & MLE Estimation ---
def estimate_target_3d_given_distances(S_hover_all, distances, params, x0_guess):
    """
    Hàm MLE của bạn (đã chỉnh sửa nhẹ để nhận đầu vào là list 3 khoảng cách)
    """
    D_meas = np.array(distances)
    S_hover = np.asarray(S_hover_all, dtype=float)

    # Negative log-likelihood
    def nll(theta):
        x_t, y_t, z_t = theta
        d_pred = np.sqrt(
            (S_hover[0, :] - x_t) ** 2
            + (S_hover[1, :] - y_t) ** 2
            + (S_hover[2, :] - z_t) ** 2
        )
        # Tổng bình phương sai số (MLE tương đương Least Squares cho nhiễu Gaussian)
        return np.sum((D_meas - d_pred) ** 2)  # / (sigma**2) có thể bỏ do không ảnh hướng vị trí tối ưu

    bounds = [
        (0.0, params["sim"]["L_x"]),
        (0.0, params["sim"]["L_y"]),
        (0.0, params["sim"]["L_z"]),
    ]

    res = minimize(
        nll,
        x0_guess,
        method="L-BFGS-B",
        bounds=bounds,
    )
    return res.x, res.fun


def solve_association_and_estimate(measurements_list, S_hover, params):
    """
    Thử mọi cách ghép cặp (Permutation) để tìm ra bộ khoảng cách thuộc về cùng 1 target.
    Sau đó ước lượng vị trí.
    """
    K = len(measurements_list[0])
    M_hover = len(measurements_list)  # Sẽ là 3

    # Danh sách lưu kết quả tốt nhất cho mỗi target
    estimated_positions = []

    # Do Stage 0 chưa có thông tin trước, ta dùng Brute-force search các hoán vị.
    # Giả sử thứ tự của list thứ 1 là chuẩn, ta hoán vị list 2 và 3 để khớp.
    # (Đây là cách tiếp cận đơn giản khi không có lịch sử bay)

    # Lấy các hoán vị có thể có của index
    perms = list(permutations(range(K)))

    best_total_nll = float('inf')
    best_estimates = []

    # Điểm khởi đầu cho MLE (giả sử là Base station)
    x0_init = [base_pos[0], base_pos[1], 0]

    # Cố định thứ tự mảng đầu tiên, thử hoán vị 2 mảng còn lại
    # Mục tiêu: Tìm bộ 3 khoảng cách (1 từ HP1, 1 từ HP2, 1 từ HP3) tạo thành 1 Target
    # Lặp qua tất cả các khả năng ghép nhóm

    # Cách 1: Thử tất cả các cách ghép nhóm (Complexity O((K!)^2) nhưng K nhỏ thì OK)
    # Ở đây ta tối ưu: Ta gán các cặp triplet và chạy MLE, bộ nào có NLL thấp nhất là bộ đúng.

    # Tạo danh sách các triplet ứng cử viên
    # Ví dụ K=2:
    # Triplet A: (HP1[0], HP2[0], HP3[0]) -> Target 1
    # Triplet B: (HP1[1], HP2[1], HP3[1]) -> Target 2

    candidates = []
    for p2 in perms:  # Hoán vị cho mảng thứ 2
        for p3 in perms:  # Hoán vị cho mảng thứ 3
            # Xây dựng giả thuyết: Target i sử dụng distances (m1[i], m2[p2[i]], m3[p3[i]])
            current_estimates = []
            current_total_nll = 0

            valid_combination = True

            for k in range(K):
                d_set = [
                    measurements_list[0][k],  # Từ HP1, lấy thứ tự gốc
                    measurements_list[1][p2[k]],  # Từ HP2, lấy theo hoán vị p2
                    measurements_list[2][p3[k]]  # Từ HP3, lấy theo hoán vị p3
                ]

                pos, nll_val = estimate_target_3d_given_distances(S_hover, d_set, params, x0_init)

                # Kiểm tra hợp lý (nếu NLL quá lớn nghĩa là không giao nhau được)
                if nll_val > 1e6:
                    valid_combination = False
                    break

                current_estimates.append(pos)
                current_total_nll += nll_val

            if valid_combination and current_total_nll < best_total_nll:
                best_total_nll = current_total_nll
                best_estimates = current_estimates

    return np.array(best_estimates).T


# Chạy giải thuật
estimated_targets = solve_association_and_estimate(all_measurements_per_hover, S_hover, params)

print(f"\nVị trí Ước lượng (Estimated Targets):\n{estimated_targets.T}")

# --- 6. Vẽ đồ thị 3D ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. Vẽ Hover Points (Màu xanh dương)
ax.scatter(S_hover[0, :], S_hover[1, :], S_hover[2, :], c='blue', marker='^', s=100, label='Hover Points (Stage 0)')
# Vẽ Base
ax.scatter(base_pos[0], base_pos[1], 0, c='black', marker='s', s=100, label='Base Station')

# 2. Vẽ True Targets (Màu đỏ)
ax.scatter(true_targets[0, :], true_targets[1, :], true_targets[2, :], c='red', marker='o', s=100, label='True Targets')

# 3. Vẽ Estimated Targets (Màu xanh lá)
ax.scatter(estimated_targets[0, :], estimated_targets[1, :], estimated_targets[2, :], c='green', marker='*', s=150,
           label='Estimated Targets')

# 4. Nối đường giữa Hover Point và Target (để minh họa sensing geometry)
for k in range(K_targets):
    for h in range(3):
        ax.plot([S_hover[0, h], true_targets[0, k]],
                [S_hover[1, h], true_targets[1, k]],
                [S_hover[2, h], true_targets[2, k]], 'gray', linestyle='--', alpha=0.3)

    # Nối True vs Estimated
    ax.plot([true_targets[0, k], estimated_targets[0, k]],
            [true_targets[1, k], estimated_targets[1, k]],
            [true_targets[2, k], estimated_targets[2, k]], 'k:', linewidth=1.5)

# Thiết lập trục và nhãn
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (Altitude)')
ax.set_title(f'Stage 0: Coarse Target Localization (K={K_targets})')
ax.legend()

# Giới hạn trục Z cho đẹp hơn
ax.set_zlim(0, 500)

plt.show()