import numpy as np
from comm_model.min_total_rate import min_total_rate
from radar_sensing_model.sensing_metric_max_crb import sensing_metric_max_crb
from main_algorithms.test_constraints import test_constraints
import gc

def optimize_ga(wp_prev, estimate_target_position, params, E_remain, N_wp):
    """
    GA-based trajectory optimization for one stage (3D)

    wp_prev: None or ndarray shape (3, N_prev)
    return: wp_new shape (3, N_stg)
    """

    # =========================
    # Local shortcuts
    # =========================
    N_stg = N_wp
    pop_size = params["sim"]["population_size"]
    max_generations = params["sim"]["max_generations"]
    tournament_k = params["sim"]["tournament_k"]
    crossover_rate = params["sim"]["crossover_rate"]
    mutation_rate = params["sim"]["mutation_rate"]
    sigma = params["sim"]["mutation_sigma"]
    Vmax = params["sim"]["V_max"]
    Tf = params["sim"]["T_f"]

    L_x = params["sim"]["L_x"]
    L_y = params["sim"]["L_y"]
    L_z = params["sim"]["L_z"]

    eta = params["sim"]["eta"]
    s_c = params["setup"]["comm_user_pos"] if "setup" in params else None

    rng = np.random.default_rng()
    bandwidth_associated = [] #Băng thông của hệ thống. Khởi tạo băng thông gốc.
    # =========================
    # Determine start position
    # =========================
    if wp_prev is None:
        start_pos = params["setup"]["base_station_pos"]
    else:
        start_pos = wp_prev[:, -1]

    # =========================
    # Fitness (HARD constraint)
    # =========================
    def get_fitness_crb_rate_m(genome):

        wp_candidate = genome.reshape(3, N_stg, order="F")

        # Hard constraints (energy, velocity, etc.)
        if test_constraints(wp_candidate, start_pos, E_remain):
            return 1e12, 1e12, 0.0

        if wp_prev is None:
            total_wp = wp_candidate
        else:
            total_wp = np.concatenate([wp_prev, wp_candidate], axis=1)

        crb_val = sensing_metric_max_crb(total_wp, estimate_target_position, params)
        rate_val = min_total_rate(
            total_wp, s_c, params, total_wp.shape[1]
        )

        fitness = eta * crb_val - (1 - eta) * rate_val
        return fitness, crb_val, rate_val

    def evaluate_population(pop):
        fitness = np.zeros(pop.shape[0])
        fitness_crb = np.zeros(pop.shape[0])
        fitness_rate = np.zeros(pop.shape[0])
        for i in range(pop.shape[0]):
            f, crb_v, rate_v = get_fitness_crb_rate_m(pop[i])
            fitness[i] = f
            fitness_crb[i] = crb_v
            fitness_rate[i] = rate_v
        return fitness, fitness_crb, fitness_rate

    # =========================
    # Initialization (3D random walk)
    # =========================
    def initialize_population_structured_3d(n_individuals=None):

        if n_individuals is None:
            n_individuals = pop_size

        pop = np.zeros((n_individuals, 3 * N_stg), dtype=np.float32)
        max_step = Vmax * Tf

        n_straight = int(0.4 * n_individuals)
        n_curve = int(0.4 * n_individuals)
        n_random = n_individuals - n_straight - n_curve

        idx = 0

        # =========================
        # 1) STRAIGHT trajectories
        # =========================
        for _ in range(n_straight):
            wp = np.zeros((3, N_stg), dtype=np.float32)
            wp[:, 0] = start_pos

            direction = rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction) + 1e-9
            step = rng.uniform(0.3 * max_step, max_step)

            for i in range(1, N_stg):
                wp[:, i] = wp[:, i - 1] + step * direction
                wp[0, i] = np.clip(wp[0, i], 0, L_x)
                wp[1, i] = np.clip(wp[1, i], 0, L_y)
                wp[2, i] = np.clip(wp[2, i], 100, L_z)

            pop[idx] = wp.flatten(order="F")
            idx += 1

        # =========================
        # 2) CURVED trajectories
        # =========================
        for _ in range(n_curve):
            wp = np.zeros((3, N_stg), dtype=np.float32)
            wp[:, 0] = start_pos

            direction = rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction) + 1e-9

            axis = rng.normal(0, 1, 3)
            axis /= np.linalg.norm(axis) + 1e-9
            curvature = rng.uniform(0.05, 0.2)

            step = rng.uniform(0.3 * max_step, max_step)

            for i in range(1, N_stg):
                direction = direction + curvature * np.cross(axis, direction)
                direction /= np.linalg.norm(direction) + 1e-9
                wp[:, i] = wp[:, i - 1] + step * direction

                wp[0, i] = np.clip(wp[0, i], 0, L_x)
                wp[1, i] = np.clip(wp[1, i], 0, L_y)
                wp[2, i] = np.clip(wp[2, i], 100, L_z)

            pop[idx] = wp.flatten(order="F")
            idx += 1

        # =========================
        # 3) RANDOM WALK (ít)
        # =========================
        for _ in range(n_random):
            wp = np.zeros((3, N_stg), dtype=np.float32)
            wp[:, 0] = start_pos

            for i in range(1, N_stg):
                step = rng.normal(0, 0.3 * max_step, 3)
                wp[:, i] = wp[:, i - 1] + step
                wp[0, i] = np.clip(wp[0, i], 0, L_x)
                wp[1, i] = np.clip(wp[1, i], 0, L_y)
                wp[2, i] = np.clip(wp[2, i], 300, L_z)

            pop[idx] = wp.flatten(order="F")
            idx += 1

        return pop

    # =========================
    # GA operators
    # =========================
    def tournament_selection(pop, fitness):
        idx = rng.integers(0, pop.shape[0], size=tournament_k)
        best = idx[np.argmin(fitness[idx])]
        return pop[best].copy()

    def SBX(p1, p2, eta_SBX=2):
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(len(p1)):
            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta_SBX + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta_SBX + 1))
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        return c1, c2

    def mutate_curve_cluster(genome):

        if rng.random() > mutation_rate:
            return

        wp = genome.reshape(3, N_stg, order="F")

        # chọn đoạn cong
        # If trajectory too short, skip mutation
        if N_stg < 6:
            return

        seg_len_max = max(4, N_stg // 3)
        seg_len = rng.integers(3, seg_len_max)

        if N_stg - seg_len <= 1:
            return

        start = rng.integers(1, N_stg - seg_len)
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis) + 1e-9
        angle = rng.uniform(-0.3, 0.3)

        for i in range(start + 1, start + seg_len):
            v = wp[:, i] - wp[:, i - 1]
            v_rot = (
                    v * np.cos(angle)
                    + np.cross(axis, v) * np.sin(angle)
                    + axis * np.dot(axis, v) * (1 - np.cos(angle))
            )
            wp[:, i] = wp[:, i - 1] + v_rot

            wp[0, i] = np.clip(wp[0, i], 0, L_x)
            wp[1, i] = np.clip(wp[1, i], 0, L_y)
            wp[2, i] = np.clip(wp[2, i], 100, L_z)

        genome[:] = wp.flatten(order="F")

    # =========================
    # GA main loop (WITH ELITISM + IMMIGRATION)
    # =========================

    elite_ratio = 0.10

    elite_size = max(1, int(elite_ratio * pop_size))
    repro_size = pop_size - elite_size

    pop = initialize_population_structured_3d()
    fitness, fitness_crb, fitness_rate = evaluate_population(pop)

    best_fitness_history = []
    best_fitness_crb_history = []
    best_fitness_rate_history = []
    log_interval = 10  # hoặc 20
    for gen in range(max_generations):
        # print(f"Số gen hiện tại là: {gen}")
        idx_sorted = np.argsort(fitness)
        pop_sorted = pop[idx_sorted]

        new_pop = []

        # --- Elitism ---
        for i in range(elite_size):
            new_pop.append(pop_sorted[i].copy())

        # --- Reproduction ---
        while len(new_pop) < elite_size + repro_size:
            p1 = tournament_selection(pop, fitness)
            p2 = tournament_selection(pop, fitness)
            c1, c2 = SBX(p1, p2)
            mutate_curve_cluster(c1)
            mutate_curve_cluster(c2)
            new_pop.append(c1)
            if len(new_pop) < elite_size + repro_size:
                new_pop.append(c2)

        pop = np.asarray(new_pop, dtype=np.float32)
        fitness, fitness_crb, fitness_rate = evaluate_population(pop)
        if gen % log_interval == 0:
            best_idx = np.argmin(fitness)
            best_genome = pop[best_idx].copy()
            # best_fitness_history.append(fitness[best_idx])
            best_fitness_crb_history.append(fitness_crb[best_idx])
            best_fitness_rate_history.append(fitness_rate[best_idx])

        if gen % 50 == 0:
            gc.collect()

        # =========================
        # Decode best solution
        # =========================
    wp_new = best_genome.reshape(3, N_stg, order="F")

    return (
        wp_new,
        bandwidth_associated,
        # best_fitness_history,
        best_fitness_crb_history,
        best_fitness_rate_history
    )