from hyperparams.optimizer_param import OptimizerParam

import tensorflow as tf


def get_optimizer():
    optimizer = tf.optimizers.Adam(learning_rate=OptimizerParam.lr,
                             beta_1=OptimizerParam.beta_1,
                             beta_2=OptimizerParam.beta_2,
                             epsilon=OptimizerParam.epsilon)

    return optimizer
