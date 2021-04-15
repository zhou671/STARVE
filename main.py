from models.model import STARVE
from utils.dataset import load_img, tensor_to_image, \
    video_to_frames, frames_to_video, make_optic_flow, init_generated_image
from utils.optimizers import get_optimizer
from utils.losses import style_content_loss, tv_loss
from hyperparams.train_param import TrainParam
from hyperparams.dataset_param import DatasetParam

import tensorflow as tf
import cv2
from tqdm import tqdm
from os import makedirs
from os.path import join, isdir, basename, splitext
import glob


def preparation():
    """
    Preparations before training begins.
    Including creating directories, convert a video into frames,
    calculate optic flows, etc.
    :return:
        None
    """
    if not isdir(TrainParam.output_dir):
        makedirs(TrainParam.output_dir)
    if not isdir(TrainParam.iter_img_dir):
        makedirs(TrainParam.iter_img_dir)
    if not isdir(TrainParam.stylized_img_dir):
        makedirs(TrainParam.stylized_img_dir)

    if DatasetParam.use_video:
        # convert video to frames
        video_to_frames(DatasetParam.video_path, TrainParam.video_frames_dir)
        if TrainParam.use_optic_flow:
            make_optic_flow(TrainParam.video_frames_dir, TrainParam.video_optic_flow_dir)

    return


def train():
    model = STARVE()

    # get style target
    style_img_path = DatasetParam.style_img_path
    style_target = model(tf.constant(load_img(style_img_path)))['style']

    # get content image path list
    if DatasetParam.use_video:
        content_img_list = glob.glob(join(TrainParam.video_frames_dir,
                                          '*.{}'.format(DatasetParam.img_fmt)))
        content_img_list.sort(key=lambda x: int(splitext(basename(x))[0]))
    else:
        content_img_list = [DatasetParam.content_img_path]

    for n_img, content_img_path in enumerate(content_img_list):
        # Call tf.function each time, or there will be
        # ValueError: tf.function-decorated function tried to create variables on non-first call
        # because of issues with lazy execution.
        # https://www.machinelearningplus.com/deep-learning/how-use-tf-function-to-speed-up-python-code-tensorflow/
        tf_train_step = tf.function(train_step)

        optimizer = get_optimizer()
        content_target = model(tf.constant(load_img(content_img_path)))['content']
        generated_image = init_generated_image(int(splitext(basename(content_img_path))[0]),
                                               is_first_frame=n_img == 0)

        pbar = tqdm(range(TrainParam.n_step))
        pbar.set_description_str('[{}/{} {}]'.format(n_img + 1,
                                                     len(content_img_list),
                                                     basename(content_img_path)))
        for step in pbar:
            tf_train_step(model, generated_image, optimizer, content_target, style_target)

            if (step + 1) % TrainParam.draw_step == 0:
                cv2.imwrite(join(TrainParam.iter_img_dir, "{}.{}"
                                 .format(step + 1, DatasetParam.img_fmt)),
                            tensor_to_image(generated_image))
        else:
            cv2.imwrite(join(TrainParam.stylized_img_dir, basename(content_img_path)),
                        tensor_to_image(generated_image))

    return


def train_step(model, generated_image, optimizer, content_target, style_target):
    """
    Each training step.
    :param model: VGG
    :param generated_image: the image that needs to update
    :param optimizer: optimizer
    :param content_target: intermediate layer outputs of the content image
    :param style_target: intermediate layer outputs of the style image
    :return:
        None
    """
    with tf.GradientTape() as tape:
        outputs = model(generated_image)
        loss = style_content_loss(outputs,
                                  style_targets=style_target,
                                  content_targets=content_target)
        loss += tv_loss(generated_image)

    grad = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=-150, clip_value_max=150))

    return


def post_process():
    if DatasetParam.use_video:
        # convert frames to videos
        frames_to_video(TrainParam.stylized_img_dir,
                        join(TrainParam.output_dir, 'stylized_{}'
                             .format(basename(DatasetParam.video_path))))

    return


if __name__ == '__main__':
    preparation()
    train()
    post_process()
