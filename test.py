import numpy as np

# ===== import hàm cần test =====
from comm_model.multi_users_data_rate import stage_comm_data
from parameters import params
# Gia lap quy dao UAV bay thang co dinh.
sim = params["sim"]
setup = params["setup"]
Bm = sim["Bm"]
Nf = 20
Sj = np.zeros((3, Nf))

Sj[0, :] = np.linspace(500, 1100, Nf)   # x
Sj[1, :] = np.linspace(600, 1300, Nf)   # y
Sj[2, :] = 100                          # z (cao độ UAV)                    # z co dinh

psi_c = stage_comm_data(Sj, Bm, params)
print(psi_c.shape)
print("Total transmitted data per CU in this stage:")
for m, val in enumerate(psi_c):
    print(f"CU {m+1}: {val:.3e} bits")

