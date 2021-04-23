from utils.dataset import video_to_frames
from hyperparams.train_param import TrainParam
from hyperparams.dataset_param import DatasetParam
from hyperparams.loss_param import LossParam

import tensorflow as tf
import tensorflow_hub as hub
import cv2
from tqdm import tqdm
import os
from os import makedirs
from os.path import join, isdir, basename, splitext
import glob
import numpy as np
import moviepy
import PIL.Image

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

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

    return


def load_img(path_to_img, do_preprocess=True, bgr=True):
    """
    Load a single image.
    :param path_to_img: image path
    :param do_preprocess: subtract ImageNet mean
    :param bgr: whether to convert channel sequence to BGR
    :return:
        img: image of 4 dimensions (1, h, w, c), range (0, 255) when `do_preprocess` is False
    """
    img = tf.io.read_file(path_to_img)  # RGB
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [DatasetParam.img_h, DatasetParam.img_w])
    img = img[tf.newaxis, :]

    return img



def train():
    os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


    style_image_path = DatasetParam.style_img_path
    style_image = load_img(style_image_path, bgr=False)
    
    # get content image path list
    if DatasetParam.use_video:
        content_img_list = glob.glob(join(TrainParam.video_frames_dir,
                                          '*.{}'.format(DatasetParam.img_fmt)))
        content_img_list.sort(key=lambda x: int(splitext(basename(x))[0]))
    else:
        content_img_list = [DatasetParam.content_img_path]
    
    for content_img_path in content_img_list:
        content_img = load_img(content_img_path, bgr=False)

        generated_image = model(tf.constant(content_img), tf.constant(style_image))[0]
        frame_idx = int(splitext(basename(content_img_path))[0]) \
                if DatasetParam.use_video else splitext(basename(content_img_path))[0]

        save_path = join(TrainParam.stylized_img_dir,
                                 "{}.{}".format(frame_idx, DatasetParam.img_fmt))
        
        im = tensor_to_image(generated_image)
        im = im.save(save_path)

    return


def frames_to_video(frame_folder, video_path, img_format=DatasetParam.img_fmt):
    """
    Convert stylized frames to a video.
    :param frame_folder: the folder with frame images
    :param video_path: path to save the output video
    :param n_pass: number of pass
    :param img_format: image format, default='jpg'
    :return:
        None
    """

    file_list = glob.glob(join(frame_folder, '*.{}'.format(img_format)))
    file_list.sort(key=lambda x: int(splitext(basename(x))[0]))
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(file_list, fps=DatasetParam.video_fps)
    clip.write_videofile(video_path)

    return

def post_process():
    """
    Convert the stylized frames of the final pass to a video.
    :return:
        None
    """
    if DatasetParam.use_video:
        # convert frames to videos
        frames_to_video(TrainParam.stylized_img_dir,
                        join(TrainParam.output_dir, 'stylized_{}'.format(basename(DatasetParam.video_path))))

    return


if __name__ == '__main__':
    preparation()
    train()
    post_process()
