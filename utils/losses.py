from hyperparams.loss_param import LossParam

import tensorflow as tf


def gram_matrix(input_tensor):
    """
    :param input_tensor: [b, h, w, c]
    :return:
        Gram matrix.
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
    :return:
        Weighted sum of style and content loss.
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
    :param warped_images: nXWxHx3 warped images, where n is length of J
                          omega(x) in equation
    :param per_pxl_weight: nXWxHx1 per_pxl_weight, c in equation
    :return:
        loss: weighted temporal loss
    """
    loss = tf.math.subtract(generated_image, warped_images)
    loss = tf.math.multiply(loss, loss)
    loss = tf.math.multiply(per_pxl_weight, loss)
    loss = LossParam.temporal_weight * tf.reduce_mean(loss)

    return loss


def tv_loss(generated_image):
    """
    Total variation loss.
    :param generated_image: generated image
    :return:
        loss: Total variation loss.
    """
    loss = LossParam.tv_weight * tf.image.total_variation(generated_image)

    return loss
