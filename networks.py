import numpy as np
import tensorflow as tf

import utils.ops as U


class Base(object):
    def __init__(self, type, name):
        self.name = name
        self._type = type

    def _build_network(self, batch_norm, noisy_layer, reuse):
        return NotImplementedError

    @property
    def global_vars(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self._type + '/' + self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self._type + '/' + self.name)


class Actor(Base):

    def __init__(self, name, obs, is_training, dimu, batch_norm=False,
                 noisy_layer=False, reuse=None):
        super(Actor, self).__init__('actor', name)

        self._obs = obs
        self._is_training = is_training
        self._dimu = dimu
        with tf.variable_scope(self.name):
            self.pi = self._build_network(batch_norm, noisy_layer, reuse)

    def _build_network(self, batch_norm, noisy_layer, reuse):
        out = self._obs

        layers = (512, 512, 512, )
        for i, layer in enumerate(layers):
            with tf.variable_scope('layer_{}'.format(i)):
                if noisy_layer:
                    out = U.noisy_layer(out, layer)
                else:
                    val = 1 / np.sqrt(out.get_shape().as_list()[1])
                    init = tf.random_uniform_initializer(-val, val)
                    out = tf.layers.dense(
                        out, layer, kernel_initializer=init,
                        bias_initializer=init, reuse=reuse)
                if batch_norm:
                    out = tf.layers.batch_normalization(
                        out, training=self._is_training, reuse=reuse)
                out = tf.nn.relu(out)

        init = tf.random_uniform_initializer(-3e-3, 3e-3)
        out = tf.layers.dense(out, self._dimu, kernel_initializer=init,
                              bias_initializer=init, name='output_layer',
                              reuse=reuse)
        out = tf.tanh(out)

        return out


class Critic(Base):

    def __init__(self, name, obs, u, is_training, batch_norm=False,
                 noisy_layer=False, reuse=None):
        super(Critic, self).__init__('critic', name)

        self._obs = obs
        self._u = u
        self._is_training = is_training
        with tf.variable_scope(self.name):
            self.Q = self._build_network(batch_norm, noisy_layer, reuse)

    def _build_network(self, batch_norm, noisy_layer, reuse):
        out = self._obs

        layers = (512, 512, 512, )
        for i, layer in enumerate(layers):
            with tf.variable_scope('layer_{}'.format(i)):
                if i == 1:
                    out = tf.concat((out, self._u), axis=1)
                if noisy_layer:
                    out = U.noisy_layer(out, layer)
                else:
                    val = 1 / np.sqrt(out.get_shape().as_list()[1])
                    init = tf.random_uniform_initializer(-val, val)
                    out = tf.layers.dense(
                        out, layer, kernel_initializer=init,
                        bias_initializer=init, reuse=reuse)
                if batch_norm:
                    out = tf.layers.batch_normalization(
                        out, training=self._is_training, reuse=reuse)
                out = tf.nn.relu(out)

        init = tf.random_uniform_initializer(-3e-3, 3e-3)
        out = tf.layers.dense(out, 1, kernel_initializer=init,
                              bias_initializer=init, name='output_layer',
                              reuse=reuse)

        return out
