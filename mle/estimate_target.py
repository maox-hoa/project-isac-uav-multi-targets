import numpy as np
from scipy.optimize import least_squares

# Hàm này được dùng ngay sau khi sense targets Mục đích để cập nhật vị trí các vật thể.
def estimate_target(s_hover, d_meas, p_init):
    """
    Estimate target position using nonlinear least squares.

    Parameters
    ----------
    s_hover : ndarray, shape (N, 3)
    d_meas : ndarray, shape (N)
    p_init : ndarray, shape (3)

    Returns
    -------
    p_hat : ndarray, shape (3)
    """

    def residuals(p):
        return np.linalg.norm(s_hover - p, axis=1) - d_meas

    res = least_squares(residuals, p_init)
    return res.x