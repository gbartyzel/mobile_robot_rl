import numpy as np
import tensorflow as tf
from networks.base import BaseNetwork

SEED = 1337


class Critic(BaseNetwork):

    def __init__(self, sess, config):
        super(Critic, self).__init__(config, 'critic')
        self.sess = sess

        self._build_network()
        self._build_target_network()
        self._build_train_method()

        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

        self.saver = tf.train.Saver(self.net_params)

    def train(self, batch_y, batch_state, batch_action):
        return self.sess.run([self.optim, self.loss], feed_dict={
            self.y: batch_y,
            self.states: batch_state,
            self.actions: batch_action
        })

    def predict(self, batch_action, batch_state):
        return self.sess.run(self.q_output, feed_dict={
            self.states: batch_state,
            self.actions: batch_action
        })

    def target_prediction(self, batch_action, batch_state):
        return self.sess.run(self.target_q_output, feed_dict={
            self.target_states: batch_state,
            self.target_actions: batch_action
        })

    def gradients(self, batch_state, batch_action):
        return self.sess.run(self.action_gradients, feed_dict={
            self.states: batch_state,
            self.actions: batch_action
        })[0]

    def update_target_network(self):
        self.sess.run(self.target_update)

    def save(self, time_stamp):
        self.saver.save(self.sess, 'saved_networks/critic/model',
                        global_step=time_stamp)

    def _build_network(self):
        self.net_params = []
        with tf.variable_scope('critic_network'):
            self.states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'input_states')
            self.actions = tf.placeholder(
                tf.float32, [None, self.action_dim], 'input_actions')

            with tf.variable_scope('layer_1'):
                w_1 = self._variable('w', [self.state_dim, self.layer_1],
                                     self.state_dim)
                b_1 = self._variable('b', [self.layer_1], self.state_dim)
                h_1 = tf.nn.relu(tf.matmul(self.states, w_1) + b_1)
                self.net_params.extend((w_1, b_1))

            with tf.variable_scope('layer_2'):
                w_2 = self._variable('w', [self.layer_1, self.layer_2],
                                     self.layer_1+self.action_dim)
                w_2_a = self._variable('w_a', [self.action_dim, self.layer_2],
                                       self.layer_1 + self.action_dim)
                b_2 = self._variable('b', [self.layer_2],
                                     self.layer_1 + self.action_dim)
                h_2 = tf.nn.relu(tf.matmul(h_1, w_2)
                                 + tf.matmul(self.actions, w_2_a) + b_2)
                self.net_params.extend((w_2, w_2_a, b_2))

            with tf.variable_scope('output_layer'):
                w_3 = tf.get_variable(
                    'w', [self.layer_2, 1], tf.float32,
                    tf.random_uniform_initializer(-3e-3, 3e-3, SEED))
                b_3 = tf.get_variable(
                    'b', [1], tf.float32,
                    tf.random_uniform_initializer(-3e-3, 3e-3, SEED))
                self.net_params.extend((w_3, b_3))
                self.q_output = tf.identity(tf.matmul(h_2, w_3) + b_3)

    def _build_target_network(self):
        with tf.variable_scope('critic_target_model'):
            self.target_states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'target_states')
            self.target_actions = tf.placeholder(
                tf.float32, [None, self.action_dim], 'target_states')

            ema = tf.train.ExponentialMovingAverage(decay=(1.-self.tau))
            self.target_update = ema.apply(self.net_params)
            self.target_net_params = [ema.average(x) for x in self.net_params]

            with tf.variable_scope('layer_1'):
                h_1 = tf.nn.relu(
                    tf.matmul(self.target_states, self.target_net_params[0])
                    + self.target_net_params[1])

            with tf.variable_scope('layer_2'):
                h_2 = tf.nn.relu(
                    tf.matmul(h_1, self.target_net_params[2])
                    + tf.matmul(self.target_actions, self.target_net_params[3])
                    + self.target_net_params[4])

            with tf.variable_scope('layer_3'):
                self.target_q_output = tf.identity(
                    tf.matmul(h_2, self.target_net_params[5])
                    + self.target_net_params[6])

    def _build_train_method(self):
        with tf.variable_scope('critic_optimizer'):
            self.y = tf.placeholder(tf.float32, [None, 1])
            weight_decay = tf.add_n(
                [0.01 * tf.nn.l2_loss(var) for var in self.net_params])
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.y, self.q_output)) + weight_decay
            self.optim = tf.train.AdamOptimizer(self.lrate).minimize(self.loss)
            self.action_gradients = tf.gradients(self.q_output, self.actions)

    def _variable(self, name, shape, fan):
        init = tf.random_uniform_initializer(
            -1/np.sqrt(fan), 1/np.sqrt(fan), SEED)
        return tf.get_variable(name, shape, tf.float32, init)
