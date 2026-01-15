import numpy as np

from parameters import params
from trajectory.calc_real_energy import calc_real_energy_3d
from trajectory.calc_velocity import calc_velocity


def generate_straight_trajectory_3d(start, direction, step, K):
    """
    Generate a straight-line 3D trajectory.

    S.shape = (3, K)
    """
    S = np.zeros((3, K))
    for k in range(K):
        S[:, k] = start + (k + 1) * step * direction
    return S


def test_hover_only():
    """
    Case 1: UAV does not move → only hover energy
    """
    K = params["sim"]["K_stg"]
    s_s = np.array([0.0, 0.0, 100.0])

    # UAV stays at the same point
    S = np.tile(s_s.reshape(3, 1), (1, K))

    E = calc_real_energy_3d(S, s_s, params)

    print("=== TEST 1: Hover only ===")
    print(f"Energy used: {E:.3f}")

    assert np.isfinite(E), "Energy is NaN or Inf"
    assert E > 0, "Energy must be positive"


def test_constant_velocity_motion():
    """
    Case 2: Straight-line motion with constant velocity
    """
    K = params["sim"]["K_stg"]
    s_s = np.array([0.0, 0.0, 100.0])

    direction = np.array([1.0, 0.0, 0.0])   # x direction
    step = 10.0                              # meters per waypoint

    S = generate_straight_trajectory_3d(
        start=s_s,
        direction=direction,
        step=step,
        K=K
    )

    V = calc_velocity(S, s_s, params)
    V_norm = np.linalg.norm(V, axis=0)

    print("=== TEST 2: Constant velocity ===")
    print(f"Velocity min / max: {V_norm.min():.2f} / {V_norm.max():.2f} m/s")

    E = calc_real_energy_3d(S, s_s, params)
    print(f"Energy used: {E:.3f}")

    assert np.all(V_norm <= 0.99 * params["energy"]["U_tip"] + 1e-6)
    assert np.isfinite(E)
    assert E > 0


def test_energy_monotonicity():
    """
    Case 3: Faster UAV → more energy
    """
    K = params["sim"]["K_stg"]
    s_s = np.array([0.0, 0.0, 100.0])
    direction = np.array([1.0, 0.0, 0.0])

    S_slow = generate_straight_trajectory_3d(s_s, direction, step=3.0, K=K)
    S_fast = generate_straight_trajectory_3d(s_s, direction, step=10.0, K=K)

    E_slow = calc_real_energy_3d(S_slow, s_s, params)
    E_fast = calc_real_energy_3d(S_fast, s_s, params)

    print("=== TEST 3: Energy monotonicity ===")
    print(f"E_slow = {E_slow:.3f}")
    print(f"E_fast = {E_fast:.3f}")

    assert E_fast > E_slow, "Faster UAV should consume more energy"


if __name__ == "__main__":
    print("\n==============================")
    print(" RUNNING ENERGY MODEL TESTS ")
    print("==============================\n")

    test_hover_only()
    test_constant_velocity_motion()
    test_energy_monotonicity()

    print("\n✅ ALL ENERGY TESTS PASSED\n")
