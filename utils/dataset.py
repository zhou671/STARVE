from hyperparams.dataset_param import DatasetParam

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.fx.crop import crop
from tqdm import tqdm
from os import makedirs
from os.path import dirname, join, isdir, basename, splitext
import glob


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
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [DatasetParam.img_h, DatasetParam.img_w])
    img = img[tf.newaxis, :]
    img *= 255

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
    tensor = tensor.numpy().astype(np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    return tensor


def prepare_short_video(src_video_path=r'../demo/mrbean.mp4'):
    """
    Make a short demo video.
    https://drive.google.com/file/d/1MLuux3dJVmiPTYge1J6xy5Fnq3Vd_yq6/view?usp=sharing
    :param src_video_path: path to video
    :return:
        None
    """
    video = mpy.VideoFileClip(src_video_path)
    video = video.subclip((4, 30), (4, 35))
    (w, h) = video.size
    # crop out black boarders
    video = crop(video, width=600, height=450, x_center=w / 2, y_center=h / 2)
    video.write_videofile(join(dirname(src_video_path), 'short_video.mp4'))

    return


def video_to_frames(src_video_path, save_folder, img_format=DatasetParam.img_fmt):
    """
    Convert a video to frames.
    :param src_video_path: video path
    :param save_folder: folder to save frames
    :param img_format: image format, default='jpg'
    :return:
        None
    """
    video = mpy.VideoFileClip(src_video_path)
    if not isdir(save_folder):
        makedirs(save_folder)
    for i, frame in tqdm(enumerate(video.iter_frames(fps=DatasetParam.video_fps, dtype="uint8"))):
        plt.imsave(join(save_folder, "{}.{}".format(i + 1, img_format)), frame)

    return


def frames_to_video(frame_folder, video_path, img_format=DatasetParam.img_fmt):
    """
    Convert frames to a video.
    :param frame_folder: the folder with frame images
    :param video_path: path to save the output video
    :param img_format: image format, default='jpg'
    :return:
        None
    """

    file_list = glob.glob(join(frame_folder, '*.{}'.format(img_format)))
    file_list.sort(key=lambda x: int(splitext(basename(x))[0]))
    duration = 1 / DatasetParam.video_fps
    clips = [mpy.ImageClip(x).set_duration(duration) for x in file_list]
    clips = mpy.concatenate_videoclips(clips, method="compose")
    clips.write_videofile(video_path, fps=DatasetParam.video_fps)

    return


# TODO: load image folder


if __name__ == '__main__':
    # video_to_frames(r'../demo/short_video.mp4', r'../output/video_frames')
    # frames_to_video(r'../output/video_frames', r'../output/test.mp4')
    pass
