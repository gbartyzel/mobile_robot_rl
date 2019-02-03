import numpy as np
import tensorflow as tf

import utility as U


class Base(object):
    def __init__(self, is_training_tf, use_noisynet, use_layer_norm):
        self._use_noisynet = use_noisynet
        self._use_layer_norm = use_layer_norm
        self._is_training_tf = is_training_tf

    def __call__(self, *args):
        return NotImplementedError

    def _build_hidden_layer(self, x, layer, activation_fn=None):
        if self._use_noisynet:
            x = U.independent_noisy_layer(x, layer, self._is_training_tf)
        else:
            init = self._ddpg_init(x)
            x = tf.layers.dense(x, layer, kernel_initializer=init, bias_initializer=init)
        if self._use_layer_norm:
            x = tf.contrib.layers.layer_norm(x)
        if activation_fn:
            return activation_fn(x)
        return x

    @staticmethod
    def _ddpg_init(x):
        val = 1 / np.sqrt(x.get_shape().as_list()[1])
        return tf.random_uniform_initializer(-val, val)


class Actor(Base):
    def __init__(self, action_dim, max_action, is_training_tf, use_layer_norm, use_noisynet):
        super(Actor, self).__init__(is_training_tf, use_noisynet, use_layer_norm)
        self._max_action = max_action
        self._action_dim = action_dim

    def __call__(self, observation_tf, name, reuse=None):
        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()
            out = observation_tf

            layers = (200, 100,)
            for i, layer in enumerate(layers):
                with tf.variable_scope('layer_{}'.format(i)):
                    out = self._build_hidden_layer(out, layer, tf.nn.relu)

            if self._use_noisynet:
                out = U.independent_noisy_layer(
                    out, self._action_dim, self._is_training_tf, name='output_layer')
            else:
                out = tf.layers.dense(out, self._action_dim, name='output_layer')
            out = tf.multiply(tf.sigmoid(out), self._max_action)

        return out


class Critic(Base):
    def __init__(self, is_training_tf, use_layer_norm, use_noisynet):
        super(Critic, self).__init__(is_training_tf, use_noisynet, use_layer_norm)

    def __call__(self, observation_tf, action_tf, name, reuse=None):
        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()
            out = observation_tf
            layers = (200, 100,)
            for i, layer in enumerate(layers):
                with tf.variable_scope('layer_{}'.format(i)):
                    if i == 1:
                        out = tf.concat((out, action_tf), axis=1)
                    out = self._build_hidden_layer(out, layer, tf.nn.relu)
            if self._use_noisynet:
                out = U.independent_noisy_layer(out, 1, self._is_training_tf, name='output_layer')
            else:
                out = tf.layers.dense(out, 1, name='output_layer')

        return out
