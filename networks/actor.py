import tensorflow as tf

from networks.base import BaseNetwork
from utils.ops import fc_layer


class Actor(BaseNetwork):
    def __init__(self, sess, env):
        super(Actor, self).__init__(sess, env, 'actor')

        self._states, \
        self._outputs, \
        self._net_params = self._build_network('actor_network')

        self._target_states,\
        self._target_output,\
        self._target_net_params = self._build_network('target_actor_network')

        self._build_train_method()

        self._target_update = self._build_update_method()

    def prediction(self, *args):
        return self.sess.run(
            self._outputs, feed_dict={
                self._states: args[0],
            })

    def target_prediction(self, *args):
        return self.sess.run(
            self._target_output, feed_dict={
                self._target_states: args[0],
            })

    def train(self, *args):
        self.sess.run(
            self._optim,
            feed_dict={
                self._states: args[0],
                self._q_gradients: args[1],
            })

    def update_target_network(self, phase='soft_copy'):
        if phase == 'copy':
            self.sess.run([
                self._target_net_params[i].assign(self._net_params[i])
                for i in range(len(self._net_params))
            ])
        else:
            self.sess.run(self._target_update)

    def _build_network(self, name):
        with tf.variable_scope(name):
            with tf.variable_scope('inputs'):
                states = tf.placeholder(tf.float32, [None, self._state_dim],
                                        'input_states')

            h_1 = fc_layer(states, 'layer_1',
                           [self._state_dim, self._layers[0]], tf.nn.relu)

            h_2 = fc_layer(h_1, 'layer_2', [self._layers[0], self._layers[1]],
                           tf.nn.relu)

            h_3 = fc_layer(h_2, 'layer_3', [self._layers[1], self._layers[2]],
                           tf.nn.relu)

            output = fc_layer(h_3, 'output_layer',
                              [self._layers[2], self._action_dim], tf.tanh,
                              3e-3)

        net_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return states, output, net_params

    def _build_train_method(self):
        with tf.variable_scope('actor_optimizer'):
            self._q_gradients = tf.placeholder(
                tf.float32, [None, self._action_dim], 'input_q_gradients')

            self.params_gradients = tf.gradients(
                self._outputs,
                self._net_params,
                -self._q_gradients,
                name='parameters_gradients')

            self._optim = tf.train.AdamOptimizer(self._lrate).apply_gradients(
                zip(self.params_gradients, self._net_params))

    def _build_update_method(self):
        with tf.variable_scope('actor_to_target'):
            return [
                self._target_net_params[i].assign(
                    tf.multiply(self._target_net_params[i], (1 - self._tau)) +
                    tf.multiply(self._tau, self._net_params[i]))
                for i in range(len(self._net_params))
            ]
