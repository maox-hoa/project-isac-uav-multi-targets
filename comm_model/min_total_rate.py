from comm_model.multi_users_data_rate import stage_comm_data
from comm_model.get_min_data import get_min_user_rate

def min_total_rate(Sj, Bm, params):
    psi_c = stage_comm_data(Sj, Bm, params)
    return get_min_user_rate(psi_c)

# Trong mô hình này, đầu vào đang là Sj trong đó Sj là quỹ đạo điểm bay của hệ thống
# Bm là băng thông của các người dùng.
# params là tham số của hệ thống.
