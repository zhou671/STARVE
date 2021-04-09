from hyperparams.model_param import ModelParam
from utils.losses import gram_matrix

import tensorflow as tf
from os.path import isfile, dirname, join


def get_model():
    """
    Load VGG19 with weights
    :return:
        model: VGG19 with weights
    """
    weight_file = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weight_path = join(dirname(__file__), weight_file)
    if isfile(weight_path):
        model = tf.keras.applications.VGG19(include_top=False, weights=weight_path)
    else:
        model = tf.keras.applications.VGG19(include_top=False)

    return model


def get_inter_model(model, layer_names):
    """
    Creates a vgg model that returns a list of intermediate output values.
    :param model: VGG19
    :param layer_names: list of layer names
    :return:
        inter_output: A model whose input is an image, outputs are intermediate layer outputs
    """
    model.trainable = False
    outputs = [model.get_layer(name).output for name in layer_names]
    inter_output = tf.keras.Model([model.input], outputs)

    return inter_output


class STARVE(tf.keras.models.Model):
    """
    STARVE: Style TrAnsfeR for VidEos
    """
    def __init__(self):
        super(STARVE, self).__init__()
        self.style_layers = ModelParam.style_layers
        self.content_layers = ModelParam.content_layers
        self.inter_model = get_inter_model(get_model(),
                                           self.style_layers + self.content_layers)

    def call(self, inputs):
        outputs = self.inter_model(inputs)
        style_outputs, content_outputs = (outputs[:len(self.style_layers)],
                                          outputs[len(self.style_layers):])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict,
                'style': style_dict}


if __name__ == '__main__':
    from utils.dataset import load_img, preprocess

    # net = get_model()
    # image = preprocess(load_img('../demo/dog.jpg'))
    # image = tf.image.resize(image, (224, 224))
    # prediction_probabilities = net(image)
    # predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    # print([(class_name, prob) for (number, class_name, prob) in predicted_top_5])

    net = STARVE()
    # image = preprocess(load_img('../demo/dog.jpg'))
    # res = net(image)
    # print(res)
