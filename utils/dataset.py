from hyperparams.dataset_param import DatasetParam

import tensorflow as tf
import numpy as np


def load_img(path_to_img, do_preprocess=True):
    """
    Load a single image.
    :param path_to_img: image path
    :param do_preprocess: subtract ImageNet mean
    :return:
        img: image of 4 dimensions, range (0, 255)
    """
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img *= 255
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [DatasetParam.img_h, DatasetParam.img_w])
    img = img[tf.newaxis, :]

    # preprocess
    if do_preprocess:
        img = preprocess(img)

    return img


def preprocess(img):
    """
    Preprocess.
    :param img: image tensor
    :return:
        Preprocessed image
    """
    return tf.keras.applications.vgg19.preprocess_input(img)


def tensor_to_image(tensor):
    """
    Get image from tensor.
    :param tensor: image tensor
    :return:
        Numpy array.
    """
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    return tensor.numpy()


# TODO: load image folder


if __name__ == '__main__':
    image = load_img(r'../demo/dog.jpg')
    print(tf.reduce_max(image))
