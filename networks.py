import numpy as np
import tensorflow as tf

import utills.opts as U


class Base(object):
    def __init__(self, type, name):
        self.name = name
        self._type = type

    def _build_network(self, layer_norm, noisy_layer, reuse):
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
    def __init__(self, name, obs, dimu, is_training, layer_norm=False,
                 noisy_layer=False, reuse=None):
        super(Actor, self).__init__('actor', name)

        self.obs = obs
        self._dimu = dimu
        self.is_training = is_training
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            self.pi = self._build_network(layer_norm, noisy_layer, reuse)

    def _build_network(self, layer_norm, noisy_layer, reuse):
        out = self.obs

        layers = (100, 80,)
        for i, layer in enumerate(layers):
            with tf.variable_scope('layer_{}'.format(i)):
                if noisy_layer:
                    out = U.noisy_layer(out, layer, self.is_training)
                else:
                    val = 1 / np.sqrt(out.get_shape().as_list()[1])
                    init = tf.truncated_normal_initializer(stddev=val)
                    out = tf.layers.dense(
                        out, layer, kernel_initializer=init,
                        bias_initializer=init, use_bias=False)
                if layer_norm:
                    out = tf.contrib.layers.layer_norm(out)
                out = tf.nn.relu(out)

        if noisy_layer:
            out = U.noisy_layer(
                out, self._dimu, self.is_training, name='output_layer')
        else:
            init = tf.random_uniform_initializer(-3e-3, 3e-3)
            out = tf.layers.dense(out, self._dimu, kernel_initializer=init,
                                  name='output_layer')
        out = tf.sigmoid(out)

        return out


class Critic(Base):
    def __init__(self, name, obs, u, is_training, layer_norm=False,
                 noisy_layer=False, reuse=None):
        super(Critic, self).__init__('critic', name)

        self.obs = obs
        self.u = u
        self.is_trainig = is_training
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            self.Q = self._build_network(layer_norm, noisy_layer, reuse)

    def _build_network(self, layer_norm, noisy_layer, reuse):
        out = self.obs

        layers = (100, 80,)
        for i, layer in enumerate(layers):
            with tf.variable_scope('layer_{}'.format(i)):
                if i == 1:
                    out = tf.concat((out, self.u), axis=1)
                if noisy_layer:
                    out = U.noisy_layer(out, layer, self.is_trainig)
                else:
                    val = 1 / np.sqrt(out.get_shape().as_list()[1])
                    init = tf.truncated_normal_initializer(stddev=val)
                    out = tf.layers.dense(
                        out, layer, kernel_initializer=init,
                        bias_initializer=init, use_bias=False)
                if layer_norm:
                    out = tf.contrib.layers.layer_norm(out)
                out = tf.nn.relu(out)

        if noisy_layer:
            out = U.noisy_layer(out, 1, self.is_trainig, name='output_layer')
        else:
            init = tf.random_uniform_initializer(-3e-3, 3e-3)
            out = tf.layers.dense(out, 1, kernel_initializer=init,
                                  name='output_layer')

        return out
