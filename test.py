import numpy as np

# ===== import hàm cần test =====
from comm_model.multi_users_data_rate import stage_comm_data, comm_fairness_metric

# ===== sửa comm_user_pos thành list =====
from parameters import params
params["setup"]["comm_user_pos"] = [
    params["setup"]["comm_user_pos"]
]

# ===== tạo trajectory giả =====
# UAV bay thẳng ở độ cao 200 m
Nf = 10
Sj = np.zeros((3, Nf))
Sj[0, :] = np.linspace(0, 1000, Nf)   # x
Sj[1, :] = np.linspace(0, 1000, Nf)   # y
Sj[2, :] = 200                        # z cố định

# ===== bandwidth allocation =====
M = len(params["setup"]["comm_user_pos"])
Bm = np.full(M, params["sim"]["B"])   # toàn bộ bandwidth cho CU

# ===== chạy test =====
psi_c = stage_comm_data(Sj, Bm, params)

print("psi_c:", psi_c)
print("shape:", psi_c.shape)

# ===== sanity checks =====
assert psi_c.shape == (M,)
assert np.all(psi_c > 0), "Data truyền phải > 0"

print("✅ stage_comm_data PASSED basic tests")
