import numpy as np
import tensorflow as tf

import utills.opts as U


class Base(object):
    def __init__(self, is_training_tf, use_noisynet, use_layer_norm):
        self._use_noisynet = use_noisynet
        self._use_layer_norm = use_layer_norm
        self._is_training_tf = is_training_tf

    def __call__(self, *args):
        return NotImplementedError

    def _build_hidden_layer(self, input, layer):
        if self._use_noisynet:
            input = U.independent_noisy_layer(input, layer, self._is_training_tf)
        else:
            val = 1 / np.sqrt(input.get_shape().as_list()[1])
            init = tf.truncated_normal_initializer(stddev=val)
            input = tf.layers.dense(
                input, layer, kernel_initializer=init, bias_initializer=init)
        if self._use_layer_norm:
            input = tf.contrib.layers.layer_norm(input)
        return tf.nn.elu(input)


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

            layers = (200, 150,)
            for i, layer in enumerate(layers):
                with tf.variable_scope('layer_{}'.format(i)):
                    out = self._build_hidden_layer(out, layer)

            if self._use_noisynet:
                out = U.independent_noisy_layer(
                    out, self._action_dim, self._is_training_tf, name='output_layer')
            else:
                init = tf.random_uniform_initializer(-3e-3, 3e-3)
                out = tf.layers.dense(
                    out, self._action_dim, kernel_initializer=init, name='output_layer')
            out = tf.multiply(tf.sigmoid(out), self._max_action)

        return out


class Critic(Base):
    def __init__(self, is_training_tf, use_layer_norm, use_noisynet):
        super(Critic, self).__init__(is_training_tf, use_noisynet, use_layer_norm)

    def __call__(self, observation_tf, action_tf, name, reuse=None):
        with tf.variable_scope(name) as vs:
            if reuse:
                vs.reuse_variables()
            out = tf.concat((observation_tf, action_tf), axis=1)
            layers = (200, 150,)
            for i, layer in enumerate(layers):
                with tf.variable_scope('layer_{}'.format(i)):
                    out = self._build_hidden_layer(out, layer)
            if self._use_noisynet:
                out = U.independent_noisy_layer(out, 1, self._is_training_tf, name='output_layer')
            else:
                init = tf.random_uniform_initializer(-3e-3, 3e-3)
                out = tf.layers.dense(out, 1, kernel_initializer=init, name='output_layer')

        return out

