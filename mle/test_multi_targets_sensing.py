import numpy as np
import matplotlib.pyplot as plt
from estimate_target import estimate_target
from multi_targets_sensing import sense_two_targets, associate_measurements


def plot_scene(s_hover, p1, p2, p1_hat, p2_hat):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        s_hover[:, 0],
        s_hover[:, 1],
        s_hover[:, 2],
        label="UAV positions",
        marker="o"
    )

    ax.scatter(*p1, marker="*", s=150, label="Target 1 (true)")
    ax.scatter(*p2, marker="*", s=150, label="Target 2 (true)")

    ax.scatter(*p1_hat, marker="x", s=120, label="Target 1 (estimated)")
    ax.scatter(*p2_hat, marker="x", s=120, label="Target 2 (estimated)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Two-target sensing + MLE localization")
    plt.show()


def test_pipeline():
    np.random.seed()

    # UAV hover positions (N,3)
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

    # True targets
    p1 = np.array([250, 600, 0])
    p2 = np.array([550, 600, 0])

    # Initial guesses
    p1_init = p1 + np.array([30, -40, 20])
    p2_init = p2 + np.array([-20, 30, -10])

    sigma = 2.0

    d_echoes = sense_two_targets(s_hover, p1, p2, sigma)
    D1, D2 = associate_measurements(s_hover, d_echoes, p1_init, p2_init)
    p1_hat = estimate_target(s_hover, D1, p1_init)
    p2_hat = estimate_target(s_hover, D2, p2_init)

    print("Target 1 true / estimated:", p1, p1_hat)
    print("Target 2 true / estimated:", p2, p2_hat)
    print("Echoes:\n", d_echoes)
    print("Associated D1:", D1)
    print("Associated D2:", D2)

    plot_scene(s_hover, p1, p2, p1_hat, p2_hat)


if __name__ == "__main__":
    test_pipeline()
