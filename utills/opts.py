import numpy as np
import tensorflow as tf


# ***********TensorFlow helper functions*********** #

def scaling(x, u_min, u_max, t_min=-1.0, t_max=1.0):
    """
    Scale tensor or ndarray to specific value range
    :param x: input tensor
    :param u_min: float, min value of input tensor or ndarray
    :param u_max: float, max value of input tensor or ndarray
    :param t_min: float, value of output tensor or ndarray
    :param t_max: float, value of output tensor or ndarray
    :return: output tensor or ndarray
    """
    if type(x) == np.ndarray:
        return (x - u_min) / (u_max - u_min) * (t_max - t_min) + t_min
    else:
        with tf.variable_scope('scaling'):
            m = tf.add(tf.multiply(tf.divide((x - u_min), (u_max - u_min)),
                                   (t_max - t_min)), t_min)
            return m


def dense(x, name, shape, weight_init=None, bias_init=None, weight_reg=None,
          bias_reg=None, activation=None, reuse=None):
    """
    Create a dense layer of shape n x m where n is shape of the input,
    m is declared with shape argument.
    :param x: input tensor
    :param name: string, name of the layer
    :param shape: integer, output shape of layer
    :param weight_init: initializer function for weights
    :param bias_init: initializer function for biases
    :param weight_reg: regularizer function for weights
    :param bias_reg: regularizer function for biases
    :param activation: activation function of the layer
    :param reuse: boolean, whether to reuse the weights of a previous layer
    by the same name
    :return: output tensor
    """
    if not name:
        name = 'DenseLayer'

    with tf.variable_scope(name, reuse=reuse):
        x_shape = x.get_shape().as_list()[1]
        w = tf.get_variable('weight', [x_shape, shape], tf.float32,
                            weight_init, weight_reg)
        b = tf.get_variable('bias', [shape], tf.float32, bias_init,
                            bias_reg)

        if activation:
            return activation(tf.add(tf.matmul(x, w), b))
        return tf.add(tf.matmul(x, w), b)


def noisy_layer(x, shape, activation=None, name=None, is_training=True,
                reuse=None):
    """
    Noisy layer for exploration https://arxiv.org/pdf/1706.10295.pdf.
    Create linear layer of shape n x m, where n is shape of input tensor,
    and m is defined by shape argument.
    :param x: input tensor
    :param shape: integer, output shape of layer
    :param activation: activation function of the layer
    :param name: string, name of the layer
    :param reuse: boolean, whether to reuse the weights of a previous layer
    by the same name
    :return: output tensor
    """
    if not name:
        name = 'NoisyLayer'

    x_shape = x.get_shape().as_list()[1]

    def noise_func(x_):
        return tf.multiply(tf.sign(x_), tf.sqrt(tf.abs(x_)))

    mu_val = 1 / np.sqrt(x_shape)
    mu_init = tf.random_uniform_initializer(-mu_val, mu_val, dtype=tf.float32)
    sigma_val = 0.5 / np.sqrt(x_shape)
    sigma_init = tf.constant_initializer(sigma_val, dtype=tf.float32)

    with tf.variable_scope(name, reuse=reuse):
        noise_i = tf.random_normal([x_shape, 1])
        noise_j = tf.random_normal([1, shape])

        with tf.variable_scope('kernel'):
            w_eps = noise_func(noise_i) * noise_func(noise_j)
            w_mu = tf.get_variable(
                'mean', [x_shape, shape], tf.float32, mu_init)
            w_sigma = tf.get_variable(
                'sigma', [x_shape, shape], tf.float32, sigma_init)
            w = tf.add(
                w_mu,
                tf.where(is_training, tf.multiply(w_sigma, w_eps), w_sigma))

        with tf.variable_scope('bias'):
            b_eps = tf.squeeze(noise_func(noise_j))
            b_mu = tf.get_variable('mean', [shape], tf.float32, mu_init)
            b_sigma = tf.get_variable('sigma', [shape], tf.float32, sigma_init)
            b = tf.add(
                b_mu,
                tf.where(is_training, tf.multiply(b_sigma, b_eps), b_sigma))

        if activation:
            output = activation(tf.add(tf.matmul(x, w), b))
        else:
            output = tf.add(tf.matmul(x, w), b)

        return output


def huber_loss(labels, predictions, delta=1.0, name=None):
    """
    Create huber loss term for training procedure
    https://en.wikipedia.org/wiki/Huber_loss.
    :param labels: the ground truth output tensor, same dimension ans
    predictions
    :param predictions: the predicted output tensor,
    :param delta: float, huber loss point where slop is changing form
    quadratic to linear
    :param name: string, name of the operation
    :return: float, loss tensor
    """
    if not name:
        name = 'HuberLoss'

    with tf.variable_scope(name):
        error = labels - predictions
        huber_term = tf.where(
            tf.abs(error) < delta, 0.5 * tf.square(error),
            delta * (tf.abs(error) - 0.5 * delta))
        loss = tf.reduce_mean(huber_term)

        return loss


def reduce_var(x, axis=None, keepdims=False, name='reduce_variance'):
    """
    Create operation in graph that calculate variance of the input tensor.
    :param x: input tensor
    :param axis: integer, the dimensions to reduce. If None (the default),
    reduces all dimensions. Must be in the range
    [-rank(input_tensor), rank(input_tensor))
    :param keepdims: boolean, f true, retains reduced dimensions with
    length 1.
    :param name: string, name of the operation
    :return: output tensor, variance
    """
    with tf.variable_scope(name):
        m = tf.reduce_mean(x, axis=axis, keepdims=keepdims)
        var = tf.reduce_mean(tf.square(x - m), axis=axis, keepdims=keepdims)
    return var


def reduce_std(x, axis=None, keepdims=False, name='reduce_std'):
    """
    Create operation in graph that calculate standard deviation of the input
    tensor.
    :param x: input tensor
    :param axis: integer, the dimensions to reduce. If None (the default),
    reduces all dimensions. Must be in the range
    [-rank(input_tensor), rank(input_tensor))
    :param keepdims: boolean, f true, retains reduced dimensions with
    length 1.
    :param name: string, name of the operation
    :return: output tensor, standard deviation
    """
    with tf.variable_scope(name):
        std = tf.sqrt(reduce_var(x, axis, keepdims))
        return std


def normalize(x, axis=None, keepdims=None, name='normalization'):
    """
    Create operation in graph that normalized the input tensor by its mean
    and standard deviation.
    :param x: input tensor
    :param axis: integer, the dimensions to reduce. If None (the default),
    reduces all dimensions. Must be in the range
    [-rank(input_tensor), rank(input_tensor))
    :param keepdims: boolean, f true, retains reduced dimensions with
    length 1.
    :param name: string, name of the operation
    :return: output tensor, normalized tensor
    """
    with tf.variable_scope(name):
        m = tf.reduce_mean(x, axis=axis, keepdims=keepdims)
        std = reduce_std(x, axis=axis, keepdims=keepdims)
        norm_x = (x - m) / std
        return norm_x
