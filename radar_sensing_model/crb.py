from radar_sensing_model.fisher_matrix_entry import fisher_matrix_entry
def crb(S_hover, s_target, params, eps=1e-12):
    """
    3D CRB for target localization (Nx3 convention).
    """

    theta_a = fisher_matrix_entry(S_hover, s_target, "theta_a", params)
    theta_b = fisher_matrix_entry(S_hover, s_target, "theta_b", params)
    theta_c = fisher_matrix_entry(S_hover, s_target, "theta_c", params)
    theta_d = fisher_matrix_entry(S_hover, s_target, "theta_d", params)
    theta_e = fisher_matrix_entry(S_hover, s_target, "theta_e", params)
    theta_f = fisher_matrix_entry(S_hover, s_target, "theta_f", params)

    numerator = (
        theta_b * theta_c + theta_a * theta_c + theta_a * theta_b
        - (theta_d**2 + theta_e**2 + theta_f**2)
    )

    denominator = (
        theta_a * theta_b * theta_c
        + 2 * theta_d * theta_e * theta_f
        - (theta_b * theta_f**2 + theta_a * theta_e**2 + theta_c * theta_d**2)
    )

    if denominator <= eps:
        return 1e12

    return numerator / denominator
