from models.model import STARVE
from utils.dataset import load_img, tensor_to_image
from utils.optimizers import get_optimizer
from utils.losses import style_content_loss
from hyperparams.train_param import TrainParam

import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import trange
import os


def make_dirs():
    if not os.path.isdir('output'):
        os.makedirs('output')
    if not os.path.isdir('output/iter_images'):
        os.makedirs('output/iter_images')


def train():
    model = STARVE()
    optimizer = get_optimizer()

    content_img_path = TrainParam.content_img_path
    style_img_path = TrainParam.style_img_path
    content_target = model(tf.constant(load_img(content_img_path)))['content']
    style_target = model(tf.constant(load_img(style_img_path)))['style']
    generated_image = tf.Variable(load_img(content_img_path))

    for step in trange(TrainParam.n_step):
        train_step(model, generated_image, optimizer, content_target, style_target)

        if (step + 1) % TrainParam.draw_step == 0:
            plt.imsave("output/iter_images/{}.jpg".format(step),
                       tensor_to_image(generated_image))
    else:
        plt.imsave("output/iter_images/final_output.jpg",
                   tensor_to_image(generated_image))

    return generated_image


@tf.function()
def train_step(model, generated_image, optimizer, content_target, style_target):
    with tf.GradientTape() as tape:
        outputs = model(generated_image)
        loss = style_content_loss(outputs,
                                  style_targets=style_target,
                                  content_targets=content_target)

    grad = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0, clip_value_max=255))


if __name__ == '__main__':
    make_dirs()
    train()
