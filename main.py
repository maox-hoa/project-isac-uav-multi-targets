import numpy as np
import parameters as p
from comm_model.multi_users_data_rate import stage_comm_data
from comm_model.get_min_data import get_min_user_rate


def min_total_rate(Sj, Bm, params):
    psi_c = stage_comm_data(Sj, Bm, params)
    return get_min_user_rate(psi_c)


def run_test():
    # 1. Load parameters
    params = p.params
    sim = params["sim"]

    # Lấy các thông số cần thiết
    N_stg = sim["N_stg"]  # 25
    Bm = sim["Bm"]

    # ==========================================
    # 2. TẠO QUỸ ĐẠO GIẢ LẬP 3D (Fix lỗi tại đây)
    # ==========================================
    # Giả sử UAV bay từ độ cao 0m lên độ cao 200m trong khi di chuyển ngang
    start_point = np.array([0, 0, 0])  # x, y, z
    end_point = np.array([1000, 1000, 200])  # x, y, z

    # Nội suy tuyến tính cho cả 3 trục
    x_coords = np.linspace(start_point[0], end_point[0], N_stg)
    y_coords = np.linspace(start_point[1], end_point[1], N_stg)
    z_coords = np.linspace(start_point[2], end_point[2], N_stg)  # Thêm trục Z

    # Gộp lại thành ma trận kích thước (3, N_stg)
    Sj = np.vstack((x_coords, y_coords, z_coords))

    print("=== CẤU HÌNH TEST 3D ===")
    print(f"Số lượng users (M): {len(Bm)}")
    print(f"Số điểm quỹ đạo (N_stg): {N_stg}")
    print(f"Shape của Sj (kỳ vọng 3, 25): {Sj.shape}")
    print("-" * 30)

    # 3. Tính toán chi tiết từng User
    try:
        rates_individual = stage_comm_data(Sj, Bm, params)

        print("\n=== KẾT QUẢ CHI TIẾT ===")
        for i, rate in enumerate(rates_individual):
            print(f"User {i + 1} Total Rate: {rate / 1e9:.4f} Gbits")

        # 4. Test hàm min_total_rate
        min_rate = min_total_rate(Sj, Bm, params)

        print("\n=== KẾT QUẢ HÀM MIN_TOTAL_RATE ===")
        print(f"Min Total Rate: {min_rate / 1e9:.4f} Gbits")

        # 5. Kiểm tra logic
        expected_min = np.min(rates_individual)
        if np.isclose(min_rate, expected_min):
            print("\n✅ TEST PASS: Hàm chạy thành công trong môi trường 3D.")
        else:
            print("\n❌ TEST FAIL: Giá trị không khớp.")

    except ValueError as e:
        print(f"\n❌ LỖI VẪN CÒN: {e}")
    except Exception as e:
        print(f"\n❌ LỖI KHÁC: {e}")


if __name__ == "__main__":
    run_test()