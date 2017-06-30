import numpy as np
import tensorflow as tf

from networks.base import BaseNetwork

SEED = 1337


class Actor(BaseNetwork):

    def __init__(self, sess, config):
        super(Actor, self).__init__(config, 'actor')
        self.sess = sess

        self._build_network()
        self._build_target_network()
        self._build_train_method()

        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

        self.saver = tf.train.Saver(self.net_params)

    def train(self, action_gradients, batch_state):
        self.sess.run(self.optim, feed_dict={
            self.action_gradients: action_gradients,
            self.states: batch_state
        })

    def update_target_network(self):
        self.sess.run(self.target_update)

    def action(self, state):
        return self.sess.run(self.output, feed_dict={
            self.states: [state]
        })[0]

    def actions(self, batch_state):
        return self.sess.run(self.output, feed_dict={
            self.states: batch_state
        })

    def target_actions(self, batch_state):
        return self.sess.run(self.target_output, feed_dict={
            self.target_states: batch_state
        })

    def load(self):
        checkpoint = tf.train.get_checkpoint_state("saved_networks/actor")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save(self, time_stamp):
        self.saver.save(self.sess, 'saved_networks/actor/model',
                        global_step=time_stamp)

    def _build_network(self):
        self.net_params = []
        with tf.variable_scope('actor_model'):
            self.states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'states')

            with tf.variable_scope('layer_1'):
                w_1 = self._variable('w', [self.state_dim, self.layer_1],
                                     self.state_dim)
                b_1 = self._variable('b', [self.layer_1], self.state_dim)
                h_1 = tf.nn.relu(tf.matmul(self.states, w_1) + b_1)
                self.net_params.extend((w_1, b_1))

            with tf.variable_scope('layer_2'):
                w_2 = self._variable('w', [self.layer_1, self.layer_2],
                                     self.layer_1)
                b_2 = self._variable('b', [self.layer_2], self.layer_1)
                h_2 = tf.nn.relu(tf.matmul(h_1, w_2) + b_2)
                self.net_params.extend((w_2, b_2))

            with tf.variable_scope('output_layer'):
                w_3 = tf.get_variable(
                    'w', [self.layer_2, self.action_dim], tf.float32,
                    tf.random_uniform_initializer(-3e-3, 3e-3, SEED))
                b_3 = tf.get_variable(
                    'b', [self.action_dim], tf.float32,
                    tf.random_uniform_initializer(-3e-3, 3e-3, SEED))
                h_3 = tf.nn.tanh(tf.matmul(h_2, w_3) + b_3)
                self.net_params.extend((w_3, b_3))
                self.output = tf.multiply(h_3, self.action_bound)

    def _build_target_network(self):
        with tf.variable_scope('actor_target_model'):
            self.target_states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'target_states')
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
                    + self.target_net_params[3])

            with tf.variable_scope('layer_3'):
                h_3 = tf.nn.tanh(
                    tf.matmul(h_2, self.target_net_params[4])
                    + self.target_net_params[5])
                self.target_output = tf.multiply(h_3, self.action_bound)

    def _build_train_method(self):
        with tf.variable_scope('actor_optimizer'):
            self.action_gradients = tf.placeholder(
                tf.float32, [None, self.action_dim])

            self.actor_gradients = tf.gradients(
                self.output, self.net_params, -self.action_gradients)

            self.optim = tf.train.AdamOptimizer(self.lrate).apply_gradients(
                zip(self.actor_gradients, self.net_params))

    def _variable(self, name, shape, fan):
        init = tf.random_uniform_initializer(
            -1/np.sqrt(fan), 1/np.sqrt(fan), SEED)
        return tf.get_variable(name, shape, tf.float32, init)
