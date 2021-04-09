class ModelParam:
    """
    Hyper-parameters for models/model.py
    """
    # content layers
    content_layers = ['block4_conv2']

    # style layers
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']