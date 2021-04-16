from hyperparams.loss_param import LossParam
from dataset import read_optic_flow, read_consistency
import cv2
import tensorflow as tf
import numpy as np

def get_optical_flow(cur_ind, prev_ind):
    """
    optical flow
    :param cur_frame: WxHxC image
    :param prev_frame: WxHxC image
    :return: WxHxC tensor for optical flow
    """
    # cur_ind, prev_ind # 改成input index. Before: cur_frame, prev_frame
    w_caret = read_optic_flow("../output/optic_flow/backward_{}_{}.flo".format(cur_ind, prev_ind))
    w = read_optic_flow("../output/optic_flow/forward_{}_{}.flo".format(prev_ind, cur_ind))
    # Use CV2
    # prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # cur = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    # inst = cv2.optflow.createOptFlow_DeepFlow()
    # # Optical flow in forward direction
    # w = inst.calc(prev, cur, None)
    # # Optical flow in backward direction
    # w_caret = inst.calc(cur, prev, None)
    # # cv2.optflow.writeOpticalFlow("dp.flo", w)
    return w, w_caret


def get_per_pxl_weight(cur_ind, prev_ind):
    """
    0 in disoccluded regions (as detected by forward-backward consistency) and at motion boundaries
    and 1 everywhere else
    see equation (5) and (6) for how disoccluded regions and motino boundaries are defined
    see equation (7) for how the weights are defined
    see equation (8) for how the input parameters are defined
    :param cur_frame: WxHxC image
    :param prev_frame: WxHxC image
    :return: WxHxC tensor for C per_pixel weight
    """
    # w, w_caret = get_optical_flow(cur_ind, prev_ind)
    # w_tilde = w*(prev_frame + w_caret*(cur_frame))
    # disoc_area = 
    c = read_consistency("../output/optic_flow/reliable_{}_{}.pgm".format(cur_ind, prev_ind), as_image=False)
    return c



def init_multipass_img(img_sqe, frame_idx, direction):
    """
    see equation (11) and (12)
    :param img_sqe: list of images, each img should be WxHxC
    :param frame_idx: the idx of the frame to be count loss for
    :param direction: passing direction, 1 if forward, -1 if backward
    :return: a WxHxC image
    """
    n = len(img_sqe)
    if direction == 1:  # if forward, i-1 to i
        prev_frame_idx = frame_idx - direction
    elif direction == -1:  # if backward, i to i+1
        prev_frame_idx = frame_idx
        frame_ind += 1
    if prev_frame_idx < 0 or prev_frame_idx > n:
        return tf.identity(img_sqe[frame_idx])
    else:
        per_pxl_weight = get_per_pxl_weight(frame_idx, prev_frame_idx)
        optical_flow = get_optical_flow(frame_idx, prev_frame_idx)
        per_pxl_weight_hat = 1 - per_pxl_weight
        return LossParam.blend_factor * tf.math.multiply(per_pxl_weight, optical_flow) + tf.math.multiply((LossParam.blend_factor * per_pxl_weight_hat  + (1 - LossParam.blend_factor)), img_sqe[frame_idx])


def gram_matrix(input_tensor):
    """
    :param input_tensor:
    :return:
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

    return result / num_locations


def style_content_loss(outputs, style_targets, content_targets):
    """
    Loss = content_weight * content_loss + style_weight * style_loss
    :param outputs: a dictionary of style and content output
    :param style_targets: extracted style features of the style image
    :param content_targets: extracted content features of the style image
    :return: combined style and content loss
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= LossParam.style_weight / len(style_outputs)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= LossParam.content_weight / len(content_outputs)
    loss = style_loss + content_loss

    return loss

def temporal_loss(generated_image, warped_images, per_pxl_weight):
    """
    see equation (7) for how this loss is defined
    :param generated_image: 1XWxHx3 image, x in equation
    :param warped_images: nXWxHx3 warped images.
           omega(x) in equation
    :param per_pxl_weight: 1XWxHx1 per_pxl_weight, c in equation
    :return:
        loss: weighted temporal loss
    """
    loss = generated_image - warped_images
    loss = tf.math.multiply(loss, loss)
    loss = tf.math.multiply(per_pxl_weight, loss)
    loss = LossParam.temporal_weight * tf.reduce_mean(loss)

    return loss

def short_or_long_term_loss(img_sqe, frame_idx, outputs, generated_image, style_targets, content_targets, warped_images, j=1):
    '''
    see equation (8)
    :param img_sqe: list of images, each img should be WxHxC
    :param frame_idx: the idx of the frame to be count loss for
    :param outputs: a dictionary of style and content output
    :param style_targets: extracted style features of the style image
    :param content_targets: extracted content features of the style image
    :param j: number of frames apart to get optical flow

    :return: combined style, content loss and temporal loss
    '''
    prev_frame_idx = frame_idx - j
    if prev_frame_idx < 0:
        return style_content_loss(outputs, style_targets, content_targets)
    per_pxl_weight = get_per_pxl_weight(frame_idx, prev_frame_idx)
    loss = style_content_loss(outputs, style_targets, content_targets) + 
           temporal_loss(generated_image[frame_idx], warped_images[prev_frame_idx]*(generated_image[prev_frame_idx]), per_pxl_weight)

def tv_loss(generated_image):
    """
    Total variation loss.
    :param generated_image: generated image
    :return:
        tv loss
    """
    loss = LossParam.tv_weight * tf.image.total_variation(generated_image)

    return loss
