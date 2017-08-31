import tensorflow as tf

from networks.base import BaseNetwork
from utils.ops import batch_norm_layer, variable

class Actor(BaseNetwork):

    def __init__(self, sess, config):
        super(Actor, self).__init__(config, 'actor', sess)

        self.states, \
        self.is_training,\
        self.output, \
        self.net_params = self._build_network('actor_network')

        self.target_states,\
        self.target_is_training,\
        self.target_output,\
        self.target_net_params = self._build_network(
            'target_actor_network')

        self._build_train_method()

        self.target_update = self._build_update_method()

        self.sess.run(tf.global_variables_initializer())

        self.update_target_network('copy')
        self.saver = self.load('actor')

    def action(self, state):
        return self.sess.run(self.output, feed_dict={
            self.states: [state],
            self.is_training: False
        })[0]

    def actions(self, batch_state):
        return self.sess.run(self.output, feed_dict={
            self.states: batch_state,
            self.is_training: True

        })

    def target_actions(self, batch_state):
        return self.sess.run(self.target_output, feed_dict={
            self.target_states: batch_state,
            self.target_is_training: True
        })

    def train(self, action_gradients, batch_state):
        self.sess.run(self.optim, feed_dict={
            self.q_gradients: action_gradients,
            self.states: batch_state,
            self.is_training: True
        })

    def update_target_network(self, phase='run'):
        if phase == 'first_run':
            self.sess.run([
                self.target_net_params[i].assign(self.net_params[i])
                for i in range(len(self.net_params))
            ])
        self.sess.run(self.target_update)

    def _build_network(self, name):
        net_params = []
        with tf.variable_scope(name):
            is_training = tf.placeholder(tf.bool, name='is_training')
            states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'input_states')
            states_bn = self._batch_norm_layer(states, is_training, tf.identity)

            with tf.variable_scope('layer_1'):
                w_1 = variable(
                    'w', [self.state_dim, self.layers['l1']], self.state_dim)
                b_1 = variable('b', [self.layers['l1']], self.state_dim)
                h_1 = tf.matmul(states_bn, w_1) + b_1
                h_1_bn = batch_norm_layer(h_1, is_training, tf.nn.relu)
                net_params.extend((w_1, b_1))

            with tf.variable_scope('layer_2'):
                w_2 = variable(
                    'w', [self.layers['l1'], self.layers['l2']],
                    self.layers['l1'])
                b_2 = variable('b', [self.layers['l2']], self.layers['l1'])
                h_2 = tf.matmul(h_1_bn, w_2) + b_2
                h_2_bn = batch_norm_layer(h_2, is_training, tf.nn.relu)
                net_params.extend((w_2, b_2))

            with tf.variable_scope('layer_3'):
                w_3 = variable(
                    'w', [self.layers['l2'], self.layers['l3']],
                    self.layers['l2'])
                b_3 = variable('b', [self.layers['l3']], self.layers['l2'])
                h_3 = tf.matmul(h_2_bn, w_3) + b_3
                h_3_bn = batch_norm_layer(h_3, is_training, tf.nn.relu)

            with tf.variable_scope('output_layer'):
                w_4 = tf.get_variable(
                    'w', [self.layers['l3'], self.action_dim], tf.float32,
                    tf.random_uniform_initializer(-3e-3, 3e-3))
                b_4 = tf.get_variable(
                    'b', [self.action_dim], tf.float32,
                    tf.random_uniform_initializer(-3e-3, 3e-3))
                output = tf.tanh(tf.matmul(h_3_bn, w_4) + b_4)
                net_params.extend((w_4, b_4))

        return states, is_training, output, net_params

    def _build_train_method(self):
        with tf.variable_scope('actor_optimizer'):
            self.q_gradients = tf.placeholder(
                tf.float32, [None, self.action_dim], 'input_q_gradients')

            self.params_gradients = tf.gradients(
                self.output, self.net_params, -self.q_gradients,
                name='parameters_gradients')

            self.optim = tf.train.AdamOptimizer(self.lrate).apply_gradients(
                zip(self.params_gradients, self.net_params))

    def _build_update_method(self):
        with tf.variable_scope('actor_to_target'):
            return [
                self.target_net_params[i].assign(
                    tf.add(tf.multiply(self.target_net_params[i],
                                       (1 - self.tau)),
                           tf.multiply(self.tau, self.net_params[i])))
                for i in range(len(self.net_params))
            ]
