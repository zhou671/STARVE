from hyperparams.dataset_param import DatasetParam
from hyperparams.train_param import TrainParam
from hyperparams.loss_param import LossParam

import tensorflow as tf
from tensorflow_addons.image import dense_image_warp
import numpy as np
import cv2
import moviepy.editor as mpy
from moviepy.video.fx.crop import crop
from tqdm import tqdm
from os import makedirs, chdir
from os.path import dirname, join, isdir, isfile, basename, splitext
import glob
from subprocess import run, Popen, PIPE


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
    img *= 255

    # preprocess
    if do_preprocess:  # RGB->BGR, subtract ImageNet mean [103.939, 116.779, 123.68] (BGR)
        img = preprocess(img)  # BGR
        if not bgr:
            img = img[..., ::-1]  # RGB
    elif bgr:
        img = img[..., ::-1]  # BGR

    return img


def preprocess(img):
    """
    Preprocess.
    :param img: image tensor
    :return:
        Preprocessed image, BGR, [0, 255] - ImageNet mean
    """
    return tf.keras.applications.vgg19.preprocess_input(img)


def tensor_to_image(tensor):
    """
    Get image from tensor.
    :param tensor: image tensor
    :return:
        Numpy array.
    """
    tensor = tensor.numpy()
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    mean = np.array([[[103.939, 116.779, 123.68]]], dtype=np.float32)
    tensor += mean
    tensor = np.clip(tensor, 0, 255)
    tensor = tensor.astype(np.uint8)

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
    if not isdir(save_folder):
        makedirs(save_folder)
    pbar = tqdm(enumerate(video.iter_frames(fps=DatasetParam.video_fps, dtype="uint8")))
    pbar.set_description_str("Convert video to frames")
    for i, frame in pbar:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imsave(join(save_folder, "{}.{}".format(i + 1, img_format)),
                   cv2.resize(frame, dsize=(DatasetParam.img_w, DatasetParam.img_h), interpolation=cv2.INTER_LINEAR))

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

    # compile consistency checker
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
    forward_flow_name = join(flow_folder, "forward_{}_{}.flo")
    backward_flow_name = join(flow_folder, "backward_{}_{}.flo")
    reliable_file_name = join(flow_folder, "reliable_{}_{}.pgm")

    # useful when `TrainParam.use_deep_matching_gpu` is True
    if DatasetParam.optic_flow_method == 'dm_df2' and TrainParam.use_deep_matching_gpu:
        dm_gpu = "opticFlow/web_gpudm_1.0_compiled/deep_matching_gpu_folder.py"  # DeepMatching GPU python script
        run(["python2", dm_gpu,
             '--frame-folder', frame_folder, '--output-folder', flow_folder,
             '--intervals', *[str(x) for x in LossParam.J],
             '-GPU', '--use_sparse', '--ngh_rad', '256'])

    # generate optic flow and calculate consistency
    for interval in LossParam.J:
        pbar = tqdm(range(len(content_img_list) - interval))
        pbar.set_description_str("Optic flow interval={}".format(interval))
        for i in pbar:
            j = i + interval
            name_i, name_j = i + 1, j + 1
            pbar.set_postfix_str("{} and {}".format(basename(content_img_list[i]),
                                                    basename(content_img_list[j])))
            # forward optic flow
            if not isfile(forward_flow_name.format(name_i, name_j)):
                make_single_optic_flow(content_img_list[i], content_img_list[j],
                                       forward_flow_name.format(name_i, name_j))

            # backward optic flow
            if not isfile(backward_flow_name.format(name_j, name_i)):
                make_single_optic_flow(content_img_list[j], content_img_list[i],
                                       backward_flow_name.format(name_j, name_i))

            # backward-forward consistency
            if not isfile(reliable_file_name.format(name_j, name_i)):
                run([checker_file, backward_flow_name.format(name_j, name_i), forward_flow_name.format(name_i, name_j),
                     reliable_file_name.format(name_j, name_i)])

            # forward-backward consistency
            if not isfile(reliable_file_name.format(name_i, name_j)):
                run([checker_file, forward_flow_name.format(name_i, name_j), backward_flow_name.format(name_j, name_i),
                     reliable_file_name.format(name_i, name_j)])

    return


def make_single_optic_flow(img1_path, img2_path, flow_path):
    """
    Generate a single optic flow file based on `DatasetParam.optic_flow_method`.
    Basename of `img1_path` is 'i.jpg'.
    Basename of `img1_path` is 'j.jpg'.
    If i < j, then `flow_path` is 'forward_i_j.flo'.
    If i > j, then `flow_path` is 'backward_i_j.flo'.
    :param img1_path: path to the first image
    :param img2_path: path to the second image
    :param flow_path: path to save the optic flow file
    :return:
        None
    """
    deep_match_file = "./opticFlow/deepmatching-static"
    deep_flow_file = "./opticFlow/deepflow2-static"

    if DatasetParam.optic_flow_method == 'dm_df2':
        if TrainParam.use_deep_matching_gpu:
            # If the `flow_path` is forward_i_j.flo,
            # the matching results are saved in forward_i_j.txt
            p = Popen(["cat", flow_path.replace('.flo', '.txt')], stdout=PIPE)
        else:
            p = Popen([deep_match_file, img1_path, img2_path], stdout=PIPE)
        p = Popen([deep_flow_file, img1_path, img2_path, flow_path, '-match'], stdin=p.stdout)
        p.communicate()

    elif DatasetParam.optic_flow_method == 'df2':
        run([deep_flow_file, img1_path, img2_path, flow_path], stdout=PIPE)

    else:
        raise ValueError("Unknown method to generate optic flow: {}"
                         .format(DatasetParam.optic_flow_method))

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


def read_optic_flow(path):
    """
    Read an optic flow file.
    https://stackovernet.xyz/cn/q/11106036
    https://github.com/youngjung/flow-python/blob/master/flowio.py
    :param path: path to .flo file
    :return:
        data: (height, width, 2), u = data[:, :, 0], v = data[:, :, 1]
    """
    with open(path, 'rb') as f:
        t = np.fromfile(f, np.float32, count=1).item()  # tag
        assert t == 202021.25, "{}: wrong tag (possibly due to big-endian machine?)".format(path)
        w = np.fromfile(f, np.int32, count=1).item()  # width
        h = np.fromfile(f, np.int32, count=1).item()  # height

        data = np.fromfile(f, np.float32)  # vector
        data = np.reshape(data, (h, w, 2))  # convert to x,y - flow

    return data


def init_generated_image(frame_idx, is_first_frame=False):
    """
    Initialize the first stylized image.
    :param frame_idx: the index of content image
    :param is_first_frame: whether this is the first frame
    :return:
        generated_image: Initialized stylized image.
    """

    if DatasetParam.init_generated_image_method == 'image' \
            or is_first_frame \
            or not TrainParam.use_optic_flow:
        content_img_path = join(TrainParam.video_frames_dir,
                                '{}.{}'.format(frame_idx, DatasetParam.img_fmt))
        generated_image = load_img(content_img_path)

    elif DatasetParam.init_generated_image_method == 'image_flow_warp':
        # https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp
        # image: The stylized image of the previous frame.
        # optic flow: The backward flow from current frame to the previous frame.
        # fix error: pip install gast==0.3.3
        # https://github.com/tensorflow/tensorflow/issues/44146#issuecomment-712798276
        stylized_img_path = join(TrainParam.stylized_img_dir,
                                 '{}.{}'.format(frame_idx - 1, DatasetParam.img_fmt))
        img = load_img(stylized_img_path)
        optic_flow_path = join(TrainParam.video_optic_flow_dir,
                               "backward_{}_{}.flo".format(frame_idx, frame_idx - 1))
        flow = read_optic_flow(optic_flow_path)
        flow = tf.convert_to_tensor(flow, dtype=tf.float32)
        flow = flow[tf.newaxis, :]
        generated_image = dense_image_warp(img, -flow)

    elif DatasetParam.init_generated_image_method == 'random':
        generated_image = tf.random.truncated_normal(
            shape=[1, DatasetParam.img_h, DatasetParam.img_w, 3],
            stddev=63.75)

    else:
        raise ValueError("Unknown method to initialize the first image: {}"
                         .format(DatasetParam.init_generated_image_method))

    generated_image = tf.cast(generated_image, tf.float32)
    generated_image = tf.Variable(generated_image)

    return generated_image


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # img = load_img(r'../output/video_frames/1.jpg', False, True)
    # flow = tf.convert_to_tensor(read_optic_flow(r'../output/optic_flow/backward_2_1.flo')[None, :], dtype=tf.float32)
    # o = dense_image_warp(img, -flow)
    # o = o.numpy()[0].astype('uint8')
    # cv2.imshow('', o)
    # cv2.waitKey(0)

    # print(o.shape, tf.reduce_max(o), tf.reduce_min(o))
    # flow_data = read_optic_flow(r'../output/optic_flow/forward_1_2.flo')
    # print(flow_data.shape)
    # print(flow_data)
    # video_to_frames(r'../demo/short_video.mp4', r'../output/video_frames')
    # frames_to_video(r'../output/video_frames', r'../output/test.mp4')
    # load_img(r'..\demo\mrbean.png')
    pass
