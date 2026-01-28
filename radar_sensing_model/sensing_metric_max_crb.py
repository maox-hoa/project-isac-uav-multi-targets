from radar_sensing_model.crb import crb
def sensing_metric_max_crb(S_hover, S_targets_hat, params):
    """
    max_k CRB_k metric (Nx3 convention)
    """

    crb_vals = [
        crb(S_hover, S_targets_hat[k], params)
        for k in range(S_targets_hat.shape[0])
    ]

    return max(crb_vals)