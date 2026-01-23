import numpy as np


def get_min_user_rate(psi_c_stage):
    """
    Tìm Rate nhỏ nhất trong số M người dùng (Hard Min).
    Dùng cho việc đánh giá hiệu năng thực tế.

    Parameters:
    ----------
    psi_c_stage : np.ndarray, shape (M,)
        Mảng chứa tổng dữ liệu (rate) của từng người dùng.

    Returns:
    -------
    float
        Giá trị rate nhỏ nhất.
    """
    # Tìm giá trị nhỏ nhất trong mảng numpy
    return np.min(psi_c_stage)


def get_smooth_min_user_rate(psi_c_stage, t=1.0):
    """
    Tìm xấp xỉ trơn của Rate nhỏ nhất (Log-Sum-Exp Approximation).
    Dùng cho hàm mục tiêu trong quá trình tối ưu (Gradient Descent) để đảm bảo tính khả vi.

    Tham chiếu: Công thức (37) trong bài báo ISAC UAV.

    Parameters:
    ----------
    psi_c_stage : np.ndarray, shape (M,)
        Mảng chứa tổng dữ liệu (rate) của từng người dùng.
    t : float
        Hệ số làm trơn (scaling factor). t càng lớn thì giá trị càng tiến gần về min thực tế,
        nhưng gradient có thể bị bão hòa. Bài báo gợi ý t > 0.

    Returns:
    -------
    float
        Giá trị xấp xỉ của min rate.
    """
    # Công thức: (-1/t) * log( sum( exp(-t * x) ) )
    # Đây là cận dưới trơn của hàm min
    return -(1.0 / t) * np.log(np.sum(np.exp(-t * psi_c_stage)))