import numpy as np
import tensorflow as tf


def fc_layer(x, name, shape, act_fn=None, val=None, xavier=False):
    if xavier:
        init = tf.contrib.layers.xavier_initializer()
    else:
        if not val:
            val = 1 / np.sqrt(shape[0])
        init = tf.random_uniform_initializer(-val, val, dtype=tf.float32)

    with tf.variable_scope(name):
        w = tf.get_variable('weight', shape, tf.float32, init)
        b = tf.get_variable('bias', [shape[1]], tf.float32, init)

        if act_fn:
            output = act_fn(tf.add(tf.matmul(x, w), b))
        else:
            output = tf.add(tf.matmul(x, w), b)

        return output


def noisy_layer(x, name, shape, act_fn=None):
    def noise_func(x_):
        return tf.multiply(tf.sign(x_), tf.sqrt(tf.abs(x_)))

    mu_val = 1 / np.sqrt(shape[0])
    mu_init = tf.random_uniform_initializer(-mu_val, mu_val, dtype=tf.float32)
    sigma_val = 0.4 / np.sqrt(shape[0])
    sigma_init = tf.constant_initializer(sigma_val, dtype=tf.float32)

    with tf.variable_scope(name):
        noise_i = tf.random_normal([shape[0], 1])
        noise_j = tf.random_normal([1, shape[1]])

        with tf.variable_scope('weight'):
            w_epsilon = noise_func(noise_i) * noise_func(noise_j)
            w_mu = tf.get_variable('w_mu', shape, tf.float32, mu_init)
            w_sigma = tf.get_variable('w_sigma', shape, tf.float32, sigma_init)
            w = tf.add(w_mu, tf.multiply(w_sigma, w_epsilon))

        with tf.variable_scope('bias'):
            b_epsilon = tf.squeeze(noise_func(noise_j))
            b_mu = tf.get_variable('b_mu', [shape[1]], tf.float32, mu_init)
            b_sigma = tf.get_variable('b_sigma', [shape[1]], tf.float32,
                                      sigma_init)
            b = tf.add(b_mu, tf.multiply(b_sigma, b_epsilon))

        if act_fn:
            output = act_fn(tf.add(tf.matmul(x, w), b))
        else:
            output = tf.add(tf.matmul(x, w), b)

        return output


def batch_norm_layer(x, train_phase, activation=None):
    return tf.contrib.layers.batch_norm(
        x,
        activation_fn=activation,
        center=True,
        scale=True,
        updates_collections=None,
        is_training=train_phase,
        reuse=None,
        decay=0.9,
        epsilon=1e-5)


def huber_loss(y, prediction, delta=1.0):
    error = y - prediction
    return tf.where(
        tf.abs(error) < delta, 0.5 * tf.square(error),
        delta * (tf.abs(error) - 0.5 * delta))
