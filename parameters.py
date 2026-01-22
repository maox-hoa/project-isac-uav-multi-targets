# parameters được tổ chức theo kiểu dict. Không dùng kiểu namespace na ná như matlab.
import numpy as np

def db2pow(x: float) -> float:
    return 10 ** (x / 10.0)

# ---------------------------------------------------------------------
# SIMULATION PARAMETERS  (dict)
# ---------------------------------------------------------------------
sim = {
    "alpha_0": db2pow(-50),
    "N_0": db2pow(-170) * 1e-3,
    "P": db2pow(20) * 1e-3,
    "Bm" : np.array([
        0.5e6,   # CU 1
        0.5e6    # CU 2
        ]),
    "H": 200,
    "L_x": 1500,
    "L_y": 1500,
    "L_z": 1500,
    "T_f": 1.5,
    "T_h": 1,
    "beta_0": db2pow(-47),
    "V_max": 30,
    "mu": 5,
    "eta": 1,
    "a": 10,
    "N_stg": 25,
    "max_generations": 7000,
    "population_size": 150,
    "mutation_rate": 0.1,
    "tournament_k": 3,
    "crossover_rate": 0.8,
    "mutation_sigma": 0.3
}

# dependent parameters
sim["K_stg"] = int(sim["N_stg"] / sim["mu"])
sim["G_p"] = 0.1 * sim["Bm"]
sim["sigma_0"] = np.sqrt(sim["Bm"] * sim["N_0"])
sim["N_final"] = sim["N_stg"]
sim["K_final"] = sim["K_stg"]

# ---------------------------------------------------------------------
# ENERGY PARAMETERS
# ---------------------------------------------------------------------
energy = {
    "P_0": 80,
    "U_tip": 120,
    "D_0": 0.6,
    "rho": 1.225,
    "P_I": 88.6,
    "v_0": 4.03,
    "s": 0.05,
    "A": 0.503,
    "E_stage_nominal": 5550
}

# ---------------------------------------------------------------------
# SETUP PARAMETERS
# ---------------------------------------------------------------------
setup = {
    "base_station_pos": np.array([250, 250, 1000]),
     "comm_user_pos": np.array([
        [750, 1250, 0],   # CU 1
        [1000, 800, 0],   # CU 2 (test)
    ]),
    "est_sense_target": np.array([749, 978, 0]),
    "sense_target_pos": np.array([1250, 1000, 0]),
    "final_pos": np.array([800, 1000]),
    "total_energy": 35e3,
    "energy_per_stage" : 8000,
}
# ---------------------------------------------------------------------
# WRAP THEM (dict of dicts)
# ---------------------------------------------------------------------
params = {
    "sim": sim,
    "energy": energy,
    "setup": setup,
}