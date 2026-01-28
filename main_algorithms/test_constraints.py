# Hàm này dùng để kiểm tra xem constraints có hợp lệ không.
# Trong bài toán gốc, các ràng buộc gồm có:
"""
(1) Ràng buộc về vị trí UAV: Bắt buộc phải nằm trên mặt phẳng độ cao H
(2) Ràng buộc về tọa độ UAV: Bắt buộc phải ở trong vùng từ 0 đến 1500 m
(3) Ràng buộc về vận tốc UAV: Vận tốc UAV không được lớn hơn vận tốc tối đa cho phép
(4) Năng lượng tiêu thụ của mỗi giai đoạn cần được kiểm tra xem có đủ để bay hết giai đoạn không. Nếu không, thực hiện bay rút gọn
"""
import numpy as np
from parameters import params
from trajectory.calc_velocity import calc_velocity
from trajectory.calc_real_energy import calc_real_energy
sim = params["sim"]
energy = params["energy"]
H = sim["H"]
# energy_per_stage = setup["energy_per_stage"]
def test_altitude(wp_candidate) -> bool:
    return np.any(wp_candidate[2, :] < H)
def constraints_velocity(wp_candidate, s_s) -> bool:
    """
    Kiểm tra ràng buộc vận tốc UAV theo chuẩn Euclid

    wp_candidate : ndarray (3, N)
    s_s          : ndarray (3,)
    return       : True nếu vi phạm, False nếu hợp lệ
    """
    # Vận tốc vector, shape (3, N)
    V = calc_velocity(wp_candidate, s_s, params)

    # Chuẩn Euclid theo từng bước thời gian
    speed = np.linalg.norm(V, axis=0)   # shape (N,)

    # Kiểm tra vượt quá V_max
    return np.any(speed > sim["V_max"])

# def constraints_zone(wp_candidate) -> bool:
#     test_zone_lower_bound = np.any(wp_candidate < 0)
#     test_zone_upper_bound = np.any(wp_candidate > sim["L_x"]) #Vì ta setup không gian vuông cho nên so sánh gộp luôn
#     test_zone_all = test_zone_lower_bound or test_zone_upper_bound
#     return test_zone_all # Trả về 1 nếu vi phạm. Trả về 0 nếu không vi phạm
def constraints_energy(wp_candidate, s_s, E_remain) -> bool:
    ener = calc_real_energy(wp_candidate, s_s, params)
    if ener > E_remain:
        return True
    else:
        return False
def test_constraints(wp_candidate, s_s, E_remain) -> bool:
    test_velocity = constraints_velocity(wp_candidate, s_s)
    test_alti = test_altitude(wp_candidate)
    test_energy = constraints_energy(wp_candidate, s_s, E_remain)
    return  test_velocity or test_energy or test_alti # Trả về 1 nếu vi phạm, trả về 0 nếu không vi phạm
# Về ràng buộc năng lượng, ràng buộc vận tốc nó đã đảm bảo điều ấy. Cho nên giai đoạn cuối cùng ta coi như bỏ để đơn giản hóa hệ thống.
# Nếu các ràng buộc bị vi phạm, trả về 1. Nếu các ràng buộc không bị vi phạm, trả về 0.