from hyperparams.loss_param import LossParam

import tensorflow as tf

def get_optical_flow(cur_frame, prev_frame):
    """
    optical flow
    :param cur_frame: WxHxC image
    :param prev_frame: WxHxC image
    :return: WxHxC tensor for optical flow
    """
    pass


def get_per_pxl_weight(cur_frame, prev_frame):
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
    pass


def init_img(img_sqe, frame_idx, direction):
    """
    see equation (11) and (12)
    :param img_sqe: list of images, each img should be WxHxC
    :param frame_idx: the idx of the frame to be count loss for
    :param direction: passing direction, either 1 or -1 
    :return: a WxHxC image
    """
    n = len(img_sqe)
    prev_frame_idx = frame_idx + direction
    if prev_frame_idx < 0 or prev_frame_idx > n:
        return tf.identity(img_sqe[frame_idx])
    else:
        per_pxl_weight = get_per_pxl_weight(img_sqe[frame_idx], img_sqe[prev_frame_idx])
        optical_flow = get_optical_flow(img_sqe[frame_idx], img_sqe[prev_frame_idx])
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
    :param outputs:
    :param style_targets:
    :param content_targets:
    :return:
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


def temporal_loss(generated_image, optical_flow, per_pxl_weight):
    """
    see equation (7) for how this loss is defined
    :param generated_image: WxHxC image, x in equation
    :param optical_flow: WxHxC optical flow, omega in equation
    :param per_pxl_weight: WxHxC per_pxl_weight, c in equation
    :return: shape() tensor for loss
    """
    loss = generated_image - optical_flow
    loss = tf.math.multiply(loss, loss)
    loss = tf.math.multiply(per_pxl_weight, loss)
    return tf.reduce_mean(loss)


def long_consistent_loss(frame_idx, img_sqe, style_targets, content_targets, direction):
    """
    see equation (9) for how this loss is defined
    :param frame_idx: the idx of the frame to be count loss for
    :param img_sqe: list of images, each img should be WxHxC 
    :param style_targets:
    :param content_targets:
    :param direction: passing direction, either 1 or -1
    :return: shape() tensor for loss
    """
    loss = style_content_loss(img_sqe[frame_idx])
    n = len(img_sqe)
    W, H, K = img_sqe[0].shape()
    accmulated_c = tf.zeros([W, H, K])
    blend_component = 0
    for j in LossParam.J:
        j = direction * j
        prev_frame_idx = frame_idx + j
        if prev_frame_idx >= 0 and prev_frame_idx < n:
            per_pxl_weight = get_per_pxl_weight(img_sqe[frame_idx], img_sqe[prev_frame_idx])
            optical_flow = get_optical_flow(img_sqe[frame_idx], img_sqe[prev_frame_idx])
            c_long = tf.math.maximum(per_pxl_weight - accmulated_c, 0)
            blend_component += temporal_loss(img_sqe[frame_idx], optical_flow, c_long)

    blend_component *= LossParam.blend_factor
    loss += blend_component
    return loss


def tv_loss(generated_image):
    """
    Total variation loss.
    :param generated_image: generated image
    :return:
        tv loss
    """
    loss = LossParam.tv_weight * tf.image.total_variation(generated_image)

    return loss
