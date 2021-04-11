from hyperparams.dataset_param import DatasetParam

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.fx.crop import crop
from moviepy.video.fx.resize import resize
from tqdm import tqdm
from os import makedirs, chdir
from os.path import dirname, join, isdir, isfile, basename, splitext
import glob
from subprocess import run, Popen, PIPE, STDOUT


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
    Video size will be (DatasetParam.img_h, DatasetParam.img_w)
    :param src_video_path: video path
    :param save_folder: folder to save frames
    :param img_format: image format, default='jpg'
    :return:
        None
    """
    video = mpy.VideoFileClip(src_video_path)
    video = resize(video, width=DatasetParam.img_w, height=DatasetParam.img_h)
    if not isdir(save_folder):
        makedirs(save_folder)
    for i, frame in tqdm(enumerate(video.iter_frames(fps=DatasetParam.video_fps, dtype="uint8"))):
        plt.imsave(join(save_folder, "{}.{}".format(i + 1, img_format)), frame)

    return


def make_optic_flow(frame_folder, flow_folder, img_format=DatasetParam.img_fmt):
    """
    Calculate forward and backward optic flow based on
    opticFlow/makeOptFLow.sh and run-deepflow.sh
    :param frame_folder: the folder of video frames
    :param flow_folder: the folder to save optic flow results
    :param img_format: image format, default='jpg'
    :return:
        None
    """
    if not isdir(flow_folder):
        makedirs(flow_folder)

    # make consistency checker
    if not isfile("./opticFlow/consistencyChecker/consistencyChecker"):
        chdir("./opticFlow/consistencyChecker")
        run(["make"])
        chdir("../../")

    # execution permission
    deep_match_file = "./opticFlow/deepmatching-static"
    deep_flow_file = "./opticFlow/deepflow2-static"
    checker_file = "./opticFlow/consistencyChecker/consistencyChecker"
    run(["chmod", "+x", deep_match_file])
    run(["chmod", "+x", deep_flow_file])
    run(["chmod", "+x", checker_file])

    # optic flow

    # frame files
    content_img_list = glob.glob(join(frame_folder, '*.{}'.format(img_format)))
    content_img_list.sort(key=lambda x: int(splitext(basename(x))[0]))
    forward_file_name = join(flow_folder, "forward_{}_{}.flo")
    backward_file_name = join(flow_folder, "backward_{}_{}.flo")
    reliable_file_name = join(flow_folder, "reliable_{}_{}.pgm")
    pbar = tqdm(range(len(content_img_list) - 1))
    pbar.set_description_str("Optic flow")
    for i in pbar:
        j = i + 1
        name_i, name_j = i + 1, j + 1
        # forward optic flow
        if not isfile(forward_file_name.format(name_i, name_j)):
            p = Popen([deep_match_file, content_img_list[i], content_img_list[j]], stdout=PIPE)
            p = Popen([deep_flow_file, content_img_list[i], content_img_list[j],
                       forward_file_name.format(name_i, name_j), '-match'], stdin=p.stdout)
            p.communicate()

        # backward optic flow
        if not isfile(backward_file_name.format(name_j, name_i)):
            p = Popen([deep_match_file, content_img_list[j], content_img_list[i]], stdout=PIPE)
            p = Popen([deep_flow_file, content_img_list[j], content_img_list[i],
                       backward_file_name.format(name_j, name_i), '-match'], stdin=p.stdout)
            p.communicate()

        # backward-forward check
        if not isfile(reliable_file_name.format(name_j, name_i)):
            run([checker_file, backward_file_name.format(name_j, name_i), forward_file_name.format(name_i, name_j),
                 reliable_file_name.format(name_j, name_i)])

        # forward-backward check
        if not isfile(reliable_file_name.format(name_i, name_j)):
            run([checker_file, forward_file_name.format(name_i, name_j), backward_file_name.format(name_j, name_i),
                 reliable_file_name.format(name_i, name_j)])

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


if __name__ == '__main__':
    # video_to_frames(r'../demo/short_video.mp4', r'../output/video_frames')
    # frames_to_video(r'../output/video_frames', r'../output/test.mp4')
    pass
