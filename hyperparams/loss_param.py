class LossParam:
    """
    Hyper-parameters for utils/losses.py
    """
    content_weight = 7e-1  # alpha
    style_weight = 5e-6  # beta
    tv_weight = 5e-2  # total variation loss weight

    temporal_weight = 2e1  # lambda
    J = [1, 2, 5]  # long-term consistency chosen frame
