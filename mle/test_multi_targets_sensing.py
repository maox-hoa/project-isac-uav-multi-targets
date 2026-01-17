import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from parameters import params
from multi_targets_sensing import sense_two_targets, associate_measurements
from estimate_target_3d import estimate_target_3d


def plot_targets_and_estimates(
    S_hover,
    p1_true, p2_true,
    p1_hat, p2_hat
):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        S_hover[0], S_hover[1], S_hover[2],
        marker="o", s=60, label="UAV hover"
    )

    ax.scatter(*p1_true, marker="*", s=160, label="Target 1 (true)")
    ax.scatter(*p2_true, marker="*", s=160, label="Target 2 (true)")

    ax.scatter(*p1_hat, marker="x", s=120, label="Target 1 (estimated)")
    ax.scatter(*p2_hat, marker="x", s=120, label="Target 2 (estimated)")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Two-target sensing and MLE localization")

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def test_two_target_pipeline():
    np.random.seed()

    # ---- Hover positions ----
    S_hover = np.array([
        [200, 300, 400, 500],
        [200, 200, 200, 200],
        [100, 150, 200, 250]
    ])

    # ---- True targets ----
    p1_true = np.array([250, 600, 0])
    p2_true = np.array([550, 600, 0])

    # ---- Initial guesses ----
    p1_init = p1_true + np.array([20, -30, 10])
    p2_init = p2_true + np.array([-25, 15, -5])

    sigma = params["sim"]["sigma_0"]

    # ---- Sense ----
    D_echoes = sense_two_targets(S_hover, p1_true, p2_true, sigma)

    # ---- Associate ----
    D1, D2 = associate_measurements(
        S_hover, D_echoes, p1_init, p2_init
    )

    # ---- Estimate ----
    p1_hat = estimate_target_3d(S_hover, D1, params, p1_init)
    p2_hat = estimate_target_3d(S_hover, D2, params, p2_init)

    print("Target 1 true / estimated:", p1_true, p1_hat)
    print("Target 2 true / estimated:", p2_true, p2_hat)

    # ---- Plot ----
    plot_targets_and_estimates(
        S_hover,
        p1_true, p2_true,
        p1_hat, p2_hat
    )


if __name__ == "__main__":
    test_two_target_pipeline()
