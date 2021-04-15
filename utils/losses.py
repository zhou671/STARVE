from hyperparams.loss_param import LossParam

import tensorflow as tf


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


def tv_loss(generated_image):
    """
    Total variation loss.
    :param generated_image: generated image
    :return:
        tv loss
    """
    loss = LossParam.tv_weight * tf.image.total_variation(generated_image)

    return loss
