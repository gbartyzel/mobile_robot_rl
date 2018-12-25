import numpy as np
import tensorflow as tf

import utills.opts as U


class Base(object):
    def __init__(self, type, name):
        self.name = name
        self._type = type

    def _build_network(self, *args):
        return NotImplementedError

    @property
    def global_vars(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self._type + '/' + self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._type + '/' + self.name)


class Actor(Base):
    def __init__(self, name, observation_tf, action_dim, max_action, is_training, layer_norm=False,
                 noisy_layer=False, reuse=None):
        super(Actor, self).__init__('actor', name)

        self.observation_tf = observation_tf
        self._action_dim = action_dim
        self.is_training = is_training
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            self.pi = self._build_network(max_action, layer_norm, noisy_layer, reuse)

    def _build_network(self, max_action, layer_norm, noisy_layer, reuse):
        out = self.observation_tf

        layers = (200, 150,)
        for i, layer in enumerate(layers):
            with tf.variable_scope('layer_{}'.format(i)):
                if noisy_layer:
                    out = U.independent_noisy_layer(out, layer, self.is_training)
                else:
                    val = 1 / np.sqrt(out.get_shape().as_list()[1])
                    init = tf.truncated_normal_initializer(stddev=val)
                    out = tf.layers.dense(
                        out, layer, kernel_initializer=init, bias_initializer=init)
                if layer_norm:
                    out = tf.contrib.layers.layer_norm(out)
                out = tf.nn.relu(out)

        if noisy_layer:
            out = U.independent_noisy_layer(
                out, self._action_dim, self.is_training, name='output_layer')
        else:
            init = tf.random_uniform_initializer(-3e-3, 3e-3)
            out = tf.layers.dense(
                out, self._action_dim, kernel_initializer=init, name='output_layer')
        out = tf.multiply(tf.sigmoid(out), max_action)

        return out


class Critic(Base):
    def __init__(self, name, observation_tf, action_tf, is_training, layer_norm=False,
                 noisy_layer=False, reuse=None):
        super(Critic, self).__init__('critic', name)

        self.obsservation_tf = observation_tf
        self.action_tf = action_tf
        self.is_trainig = is_training
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            self.Q = self._build_network(layer_norm, noisy_layer, reuse)

    def _build_network(self, layer_norm, noisy_layer, reuse):
        out = self.obsservation_tf

        layers = (200, 150,)
        for i, layer in enumerate(layers):
            with tf.variable_scope('layer_{}'.format(i)):
                if i == 1:
                    out = tf.concat((out, self.action_tf), axis=1)
                if noisy_layer:
                    out = U.independent_noisy_layer(out, layer, self.is_trainig)
                else:
                    val = 1 / np.sqrt(out.get_shape().as_list()[1])
                    init = tf.truncated_normal_initializer(stddev=val)
                    out = tf.layers.dense(
                        out, layer, kernel_initializer=init, bias_initializer=init)
                if layer_norm:
                    out = tf.contrib.layers.layer_norm(out)
                out = tf.nn.relu(out)

        if noisy_layer:
            out = U.independent_noisy_layer(out, 1, self.is_trainig, name='output_layer')
        else:
            init = tf.random_uniform_initializer(-3e-3, 3e-3)
            out = tf.layers.dense(out, 1, kernel_initializer=init, name='output_layer')

        return out
