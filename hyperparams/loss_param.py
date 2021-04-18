class LossParam:
    """
    Hyper-parameters for utils/losses.py
    """
    content_weight = 7e-1  # alpha
    style_weight = 5e-6  # beta
    tv_weight = 5e-2  # total variation loss weight

    temporal_weight = 2e1  # lambda
    J = [1, 2, 5]  # long-term consistency chosen frame

    use_temporal_pass = 2  # from which pass to use short-term temporal loss
    blend_factor = 0.5  # delta

    debug_loss = True  # when False, will run 1.5~2x faster
