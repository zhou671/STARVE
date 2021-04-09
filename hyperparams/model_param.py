class ModelParam:
    """
    Hyper-parameters for models/model.py
    """
    model_param_path = "models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"

    # content layers
    content_layers = ['block4_conv2']

    # style layers
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']