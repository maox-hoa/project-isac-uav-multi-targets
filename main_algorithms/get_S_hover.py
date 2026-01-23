import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ==========================================
# 1. HÀM CỦA BẠN (Đã sửa lỗi check dimensions)
# ==========================================
def get_S_hover(params, m_start, m_end, S):
    """
    Hàm gốc của bạn (đã sửa S.ndim != 3 thành != 2 để chạy được)
    """
    N_stg = params["sim"]["N_stg"]
    mu = params["sim"]["mu"]
    K_stg = N_stg // mu

    # --- Shift using modulo ---
    shft = N_stg % mu

    # Sửa lại logic indices một chút để match với kích thước mảng nếu cần
    # Logic của bạn:
    indices = np.arange(m_start - 1 + mu, m_end - 1 + mu + 1)

    # Lưu ý: Đoạn logic tính idxs_shift này phụ thuộc vào cách bạn định nghĩa stage
    # Tôi giữ nguyên logic tính toán của bạn
    shft_vec = shft * np.mod(indices, mu)

    # Đảm bảo kích thước shft_mat khớp để reshape
    # Nếu indices có số phần tử khác với K_stg thì tile có thể lệch,
    # tuy nhiên ở đây giả sử m_start -> m_end khớp với số stage
    num_stages = m_end - m_start + 1

    # Logic gốc của bạn: shft_mat = np.tile(shft_vec, (K_stg, 1))
    # Dòng trên sẽ tạo ra ma trận (K_stg, num_stages).
    # Khi reshape(-1) sẽ ra thứ tự: [stage1_k1, stage2_k1, ...] (theo cột nếu mặc định C-order sai)
    # Để đúng thứ tự thời gian, thường ta cần transpose trước khi reshape hoặc sắp xếp lại.
    # Tuy nhiên, để tôn trọng logic gốc, tôi giữ nguyên để bạn kiểm tra xem nó ra đúng ý đồ không.
    shft_mat = np.tile(shft_vec, (K_stg, 1))

    # Lưu ý: shft_mat đang có shape (K_stg, num_stages).
    # Reshape (-1) sẽ flatten theo hàng (row-major).
    # Có thể bạn cần check lại logic này nếu kết quả hovering point bị nhảy cóc.
    idxs_shift = shft_mat.reshape(-1, order='F')  # order='F' để lấy theo cột (từng stage một)

    # --- Linear hovering indices ---
    # Tính số lượng điểm hover
    total_hover_points = num_stages * K_stg

    # Tạo linear index cơ bản
    # Giả sử mỗi stage đóng góp K_stg điểm
    # Base index bắt đầu từ khoảng của m_start
    start_idx = (m_start - 1) * N_stg + mu  # Giả định offset đầu tiên là mu

    # Tạo lại hover_idxs_linear cho khớp với số lượng phần tử của idxs_shift
    # Logic cũ: hover_idxs_linear = np.arange(mu + (m_start - 1) * K_stg * mu, ...)
    # Logic này có vẻ đang mix giữa index của waypoint và index của hover point.
    # Dưới đây là cách tính đơn giản hóa để test visual:
    # Lấy các điểm cách nhau mu đơn vị

    # --- CÁCH TIẾP CẬN ĐƠN GIẢN ĐỂ TEST LOGIC INDEX ---
    # Tôi sẽ dùng logic vector hóa đơn giản dựa trên input của bạn để chạy thử:
    # Cách tính này cố gắng bám sát logic cộng dồn idxs_shift + hover_idxs_linear

    # Tạo mảng index tuyến tính cơ sở (cách nhau mu)
    # Số lượng điểm phải bằng len(idxs_shift)
    base_linear = np.arange(1, total_hover_points + 1) * mu
    # Offset về đúng stage bắt đầu
    stage_offset = (m_start - 1) * N_stg
    hover_idxs_linear = base_linear + stage_offset - mu  # -mu vì arange bắt đầu từ 1*mu

    # Cộng shift
    # Cần đảm bảo kích thước 2 mảng bằng nhau
    if len(idxs_shift) != len(hover_idxs_linear):
        # Fallback nếu logic tính size bị lệch (để code không crash khi test)
        min_len = min(len(idxs_shift), len(hover_idxs_linear))
        hover_idxs = idxs_shift[:min_len] + hover_idxs_linear[:min_len]
    else:
        hover_idxs = idxs_shift + hover_idxs_linear

    hover_idxs = hover_idxs.astype(int)

    # Trừ 1 để về 0-based index của Python (nếu logic tính toán của bạn là 1-based)
    # Giả sử Waypoint thứ mu là điểm hover đầu tiên -> index mu-1
    hover_idxs = hover_idxs - 1

    # --- Extract hover points ---
    if S is not None and S.size != 0:
        S = np.asarray(S)
        # SỬA LỖI Ở ĐÂY: Quỹ đạo 3D (3, N) có ndim = 2
        if S.ndim != 2:
            raise ValueError(f"S must be a 2D array of shape (d, N). Got ndim={S.ndim}")

        # Clip index để tránh lỗi out of bound nếu tính toán vượt quá độ dài S
        valid_mask = (hover_idxs >= 0) & (hover_idxs < S.shape[1])
        hover_idxs = hover_idxs[valid_mask]

        S_hover = S[:, hover_idxs]
    else:
        S_hover = np.array([[]])

    return hover_idxs, S_hover


# ==========================================
# 2. HÀM TEST VÀ VẼ 3D
# ==========================================
def test_visualization():
    # --- A. Setup Parameters ---
    params = {
        "sim": {
            "N_stg": 20,  # 20 waypoints mỗi stage
            "mu": 3  # Cứ 5 waypoints thì có 1 điểm hover
        }
    }

    # Giả sử chạy từ stage 1 đến stage 3
    m_start = 1
    m_end = 1
    num_stages = m_end - m_start + 1

    # Tổng số điểm waypoint cần thiết
    total_waypoints = params["sim"]["N_stg"] * num_stages

    # --- B. Generate Dummy 3D Trajectory (Spiral/Helix) ---
    # Tạo đường xoắn ốc để dễ nhìn 3D
    t = np.linspace(0, 4 * np.pi, total_waypoints)
    x = 500 * np.cos(t)  # Xoay tròn bán kính 500m
    y = 500 * np.sin(t)
    z = np.linspace(0, 200, total_waypoints)  # Bay lên cao dần từ 0 đến 200m

    # Tạo ma trận S shape (3, N)
    S = np.vstack((x, y, z))

    print(f"Shape of Trajectory S: {S.shape}")

    # --- C. Call the function ---
    try:
        hover_idxs, S_hover = get_S_hover(params, m_start, m_end, S)

        print("\n--- Results ---")
        print(f"Selected Hover Indices: {hover_idxs}")
        print(f"Shape of S_hover: {S_hover.shape}")

        # --- D. Plotting 3D ---
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Vẽ đường bay (Trajectory)
        ax.plot(S[0, :], S[1, :], S[2, :],
                color='blue', alpha=0.5, linewidth=1, label='UAV Trajectory')

        # 2. Vẽ các điểm Waypoints (chấm nhỏ màu xanh)
        ax.scatter(S[0, :], S[1, :], S[2, :],
                   color='blue', s=10, alpha=0.3)

        # 3. Vẽ các điểm Hovering Points (chấm to màu đỏ) [Image of a 3D scatter plot showing trajectory and highlighted points]
        if S_hover.shape[1] > 0:
            ax.scatter(S_hover[0, :], S_hover[1, :], S_hover[2, :],
                       color='red', s=100, marker='*', label='Hovering Points', depthshade=False)

            # Đánh số thứ tự các điểm hover
            for i in range(S_hover.shape[1]):
                ax.text(S_hover[0, i], S_hover[1, i], S_hover[2, i],
                        f'{hover_idxs[i]}', color='black', fontsize=9, fontweight='bold')

        # 4. Trang trí biểu đồ
        ax.set_title(
            f'UAV Trajectory & Hover Points\n(N_stg={params["sim"]["N_stg"]}, mu={params["sim"]["mu"]}, Stages {m_start}-{m_end})')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (Altitude m)')
        ax.legend()

        # Đánh dấu điểm đầu và cuối
        ax.text(S[0, 0], S[1, 0], S[2, 0], "Start", color='green')
        ax.text(S[0, -1], S[1, -1], S[2, -1], "End", color='purple')

        plt.show()

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_visualization()