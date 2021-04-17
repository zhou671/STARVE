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
        cv2.imwrite(join(save_folder, "{}.{}".format(i + 1, img_format)),
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


def read_consistency(path="", as_image=False):
    """
    Read a consistency file.
    :param path: path to .pgm file
    :param as_image: whether to return an image
    :return:
        data: When `as_image` is True, (height, width), range [0, 255]
              When `as_image` is False, (1, height, width, 1), range [0, 1]
    """
    data = cv2.imread(path, -1)
    if as_image:
        return data

    data = tf.cast(data, tf.float32) / 255.
    data = data[tf.newaxis, ..., tf.newaxis]

    return data


def read_single_consistency(frame_idx, step=1, n_pass=1):
    """
    Read a consistency file. The path is deduced from arguments.
    :param frame_idx: the index of content image, i.e., frame file name is `frame_idx`.jpg.
    :param step: frame index difference
    :param n_pass: number of pass
    :return:
        consistency: (1, height, width, 1), range [0, 1].
    """
    last_idx = frame_idx - step if n_pass % 2 == 1 else frame_idx + step
    consistency_path = join(TrainParam.video_optic_flow_dir,
                            "reliable_{}_{}.pgm".format(frame_idx, last_idx))

    consistency = read_consistency(consistency_path)

    return consistency


def init_generated_image(frame_idx, n_pass=1, is_start=False):
    """
    Initialize the first stylized image.
    :param frame_idx: the index of content image, i.e., frame file name is `frame_idx`.jpg.
    :param n_pass: number of pass
    :param is_start: useful when using multi-pass
                     True: the first frame in forward pass, or the last frame for the backward pass
                     False: otherwise
    :return:
        generated_image: Initialized stylized image.
    """
    if TrainParam.n_passes == 1 or n_pass == 1:
        if DatasetParam.init_generated_image_method == 'image' \
                or (frame_idx == 1 and DatasetParam.init_generated_image_method == 'image_flow_warp'):
            content_img_path = join(TrainParam.video_frames_dir,
                                    '{}.{}'.format(frame_idx, DatasetParam.img_fmt))
            generated_image = load_img(content_img_path)

        elif DatasetParam.init_generated_image_method == 'image_flow_warp':
            # https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp
            # image: The stylized image of the previous frame.
            # optic flow: The backward flow from current frame to the previous frame.
            # fix error: pip install gast==0.3.3
            # https://github.com/tensorflow/tensorflow/issues/44146#issuecomment-712798276
            assert not TrainParam.use_optic_flow, \
                "You are not using optic flow. You set TrainParam.use_optic_flow={}"\
                .format(TrainParam.use_optic_flow)
            generated_image = warp_single_image(frame_idx, 1)

        elif DatasetParam.init_generated_image_method == 'random':
            generated_image = tf.random.truncated_normal(
                shape=[1, DatasetParam.img_h, DatasetParam.img_w, 3],
                stddev=63.75)

        else:
            raise ValueError("Unknown method to initialize the first image: {}"
                             .format(DatasetParam.init_generated_image_method))

    else:
        stylized_img_last_pass_path = join(TrainParam.stylized_img_dir,
                                           "{}_p{}.{}".format(frame_idx, n_pass - 1, DatasetParam.img_fmt))
        stylized_img_last_pass = load_img(stylized_img_last_pass_path)
        if is_start:
            generated_image = stylized_img_last_pass
        else:
            warped_image_this_pass = warp_single_image(frame_idx, 1, n_pass)
            weight = read_single_consistency(frame_idx, 1, n_pass) * LossParam.blend_factor
            generated_image = weight * warped_image_this_pass + (1 - weight) * stylized_img_last_pass

    generated_image = tf.cast(generated_image, tf.float32)
    generated_image = tf.Variable(generated_image)

    return generated_image


def warp_single_image(frame_idx, step=1, n_pass=1):
    """
    Warp a single stylized image based on optic flow.
    The stylized image is `frame_idx`.jpg.
    The optic flow is backward_`frame_idx`_`frame_idx - step`.flo
    :param frame_idx: the index of content image, i.e., frame file name is `frame_idx`.jpg.
    :param step: frame index difference
    :param n_pass: number of pass
    :return:
        warped_image: Of shape (1, h, w, 3), subtracted by ImageNet mean.
    """
    if n_pass % 2 == 1:  # forward pass
        stylized_img_path = join(TrainParam.stylized_img_dir,
                                 '{}_p{}.{}'.format(frame_idx - step, n_pass, DatasetParam.img_fmt))
        optic_flow_path = join(TrainParam.video_optic_flow_dir,
                               "backward_{}_{}.flo".format(frame_idx, frame_idx - step))

    else:  # backward pass
        stylized_img_path = join(TrainParam.stylized_img_dir,
                                 '{}_p{}.{}'.format(frame_idx + step, n_pass, DatasetParam.img_fmt))
        optic_flow_path = join(TrainParam.video_optic_flow_dir,
                               "forward_{}_{}.flo".format(frame_idx, frame_idx + step))

    img = load_img(stylized_img_path)
    flow = read_optic_flow(optic_flow_path)
    flow = tf.convert_to_tensor(flow, dtype=tf.float32)
    flow = flow[tf.newaxis, :]
    warped_image = dense_image_warp(img, -flow)

    return warped_image


def make_warped_images_for_temporal_loss(frame_idx, n_pass=1, is_start=False):
    """
    Get warped images for short/long term temporal loss.
    :param frame_idx: the index of content image, i.e., frame file name is `frame_idx`.jpg
    :param n_pass: number of pass
    :param is_start: useful when using multi-pass
                     True: the first frame in forward pass, or the last frame for the backward pass
                     False: otherwise
    :return:
        warped_images: Of shape (b, h, w, 3), 0 <= b <= len(J).
                       Subtracted by ImageNet mean.
    """
    if TrainParam.n_passes == 1:  # no multi-pass
        warped_images = [tf.squeeze(warp_single_image(frame_idx, j), [0])
                         for j in LossParam.J
                         if frame_idx - j > 0]
    else:  # multi-pass
        if LossParam.use_temporal_pass < n_pass or is_start:
            return None
        warped_images = warp_single_image(frame_idx, 1, n_pass)  # only use short-term loss

    if not warped_images:
        return None
    warped_images = tf.convert_to_tensor(warped_images, dtype=tf.float32)
    warped_images = tf.constant(warped_images, dtype=tf.float32)

    return warped_images


def make_consistency_for_temporal_loss(frame_idx, n_pass=1, is_start=False):
    """
    Get warped images for short/long term temporal loss.
    :param frame_idx: the index of content image, i.e., frame file name is `frame_idx`.jpg
    :param n_pass: number of pass
    :param is_start: useful when using multi-pass
                     True: the first frame in forward pass, or the last frame for the backward pass
                     False: otherwise
    :return:
        consistency_weights: Of shape (b, h, w, 1), 0 <= b <= len(J).
                             Of range [0, 1].
    """
    if TrainParam.n_passes == 1:  # no multi-pass
        consistency_weights = [tf.squeeze(read_single_consistency(frame_idx, j), [0])
                               for j in LossParam.J
                               if frame_idx - j > 0]
    else:  # multi-pass
        if LossParam.use_temporal_pass < n_pass or is_start:
            return None
        consistency_weights = read_single_consistency(frame_idx, 1, n_pass)  # only use short-term loss

    if not consistency_weights:
        return None
    consistency_weights = tf.convert_to_tensor(consistency_weights, dtype=tf.float32)

    if consistency_weights.shape[0] > 1:
        consistency_weights -= tf.cumsum(consistency_weights, axis=0, exclusive=True)
        consistency_weights = tf.maximum(consistency_weights, 0)
    consistency_weights = tf.constant(consistency_weights)

    return consistency_weights


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.chdir('..')

    print(make_consistency_for_temporal_loss(2))
    # data = read_consistency(r'../output/optic_flow/reliable_2_3.pgm')
    # cv2.imshow('', data)
    # cv2.waitKey(0)
    # print(data.shape, data.max(), data.min())

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
