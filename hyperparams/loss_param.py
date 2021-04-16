class LossParam:
    """
    Hyper-parameters for utils/losses.py
    """
    content_weight = 1e-1 # alpha
    style_weight = 5e-6 # beta
    tv_weight = 1e-1  # total variation loss weight
    temporal_weight = 200  #lambda
    blend_factor = 0.5 # delta
    J = [1, 10, 20, 40] # long-term consistency chosen frame
