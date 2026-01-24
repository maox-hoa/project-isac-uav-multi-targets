import numpy as np
import parameters as p
from stage_0_initialization import run_stage_0

# Load params từ file parameters.py của bạn
params = p.params

# Định nghĩa các Targets thật (Ground Truth)
# Lưu ý: targets thật cần nằm ở vị trí mà 3 điểm quanh base có thể nhìn thấy rõ
true_targets = np.array([
    [600, 600, 0],   # Target A
    [900, 400, 0],   # Target B (Khác A)
])

# Chạy thử
estimated_targets = run_stage_0(params, true_targets)

# Đánh giá sai số
print("\n--- SO SÁNH VỚI GROUND TRUTH ---")
# Vì thứ tự output có thể bị đảo (do ta không biết ai là ai),
# ta cần tìm cặp gần nhất để so sánh lỗi.
for t_true in true_targets:
    dists = [np.linalg.norm(t_true - t_est) for t_est in estimated_targets]
    min_err = min(dists)
    print(f"Target True {t_true} -> Min Error: {min_err:.2f} meters")