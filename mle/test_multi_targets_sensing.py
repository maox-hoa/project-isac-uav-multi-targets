import numpy as np
import matplotlib.pyplot as plt
from parameters import params
from estimate_target import estimate_target   # phiên bản 1-target MLE
from multi_targets_sensing import sense_two_targets, associate_measurements


def plot_scene(s_hover, P_true, P_hat):
    """
    Plot UAV trajectory, true targets and estimated targets.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        s_hover[:, 0],
        s_hover[:, 1],
        s_hover[:, 2],
        label="UAV positions",
        marker="o"
    )

    # True targets
    ax.scatter(*P_true[:, 0], marker="*", s=150, label="Target 1 (true)")
    ax.scatter(*P_true[:, 1], marker="*", s=150, label="Target 2 (true)")

    # Estimated targets
    ax.scatter(*P_hat[:, 0], marker="x", s=120, label="Target 1 (estimated)")
    ax.scatter(*P_hat[:, 1], marker="x", s=120, label="Target 2 (estimated)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Two-target sensing + MLE localization")
    ax.legend()
    plt.show()


def test_pipeline():
    np.random.seed()

    # -------------------------------------------------
    # UAV hover positions (N, 3)
    # -------------------------------------------------
    s_hover = np.array([
        [200, 200, 100],
        [300, 250, 150],
        [400, 500, 200],
        [500, 700, 200],
        [600, 800, 200],
        [700, 100, 200],
        [800, 1000, 200],
        [900, 290, 200],
        [1000, 1000, 200],
        [1100, 800, 200],
        [1200, 900, 250]
    ])

    # -------------------------------------------------
    # True target positions (3,2)
    # -------------------------------------------------
    P_true = np.array([
        [250, 550],
        [600, 600],
        [0,   0]
    ])

    # -------------------------------------------------
    # Initial guesses (3,2)
    # -------------------------------------------------
    P_init = np.zeros((3, 2))
    P_init[:, 0] = P_true[:, 0] + np.array([30, -40, 20])
    P_init[:, 1] = P_true[:, 1] + np.array([-20, 30, -10])

    sigma = 2.0

    # -------------------------------------------------
    # Sensing (unlabeled echoes)
    # -------------------------------------------------
    d_echoes = sense_two_targets(s_hover, P_true, sigma)

    # -------------------------------------------------
    # Association
    # -------------------------------------------------
    D_assoc = associate_measurements(
        s_hover,
        d_echoes,
        P_init
    )   # shape (2, N)

    # -------------------------------------------------
    # Estimation (MLE per target)
    # -------------------------------------------------
    P_hat = np.zeros((3, 2))
    P_hat[:, 0] = estimate_target(
        s_hover,
        D_assoc[0, :],
        params,
        x0 = None
    )
    P_hat[:, 1] = estimate_target(
        s_hover,
        D_assoc[1, :],
        params,
        x0 = None
    )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------
    print("Target 1 true / estimated:")
    print(P_true[:, 0], P_hat[:, 0])

    print("Target 2 true / estimated:")
    print(P_true[:, 1], P_hat[:, 1])

    print("Echoes:\n", d_echoes)
    print("Associated D (target 1):", D_assoc[0])
    print("Associated D (target 2):", D_assoc[1])

    plot_scene(s_hover, P_true, P_hat)


if __name__ == "__main__":
    test_pipeline()
