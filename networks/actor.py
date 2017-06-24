import tensorflow as tf
from base import BaseNetwork


class Actor(BaseNetwork):

    def __init__(self, sess, config):
        super(Actor, self).__init__(config, 'actor')
        self.sess = sess

        self._build_network()
        self._build_target_network()
        self._build_train_method()

        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

    def _build_network(self):
        w_init = tf.random_normal_initializer(stddev=0.01)
        b_init = tf.zeros_initializer(0.0)
        self.net_params = []
        with tf.variable_scope('actor_model'):
            self.states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'states')

            with tf.variable_scope('layer_1'):
                w_1 = tf.get_variable(
                    'w', [self.state_dim, self.layer_1], tf.float32, w_init)
                b_1 = tf.get_variable(
                    'b', [self.layer_1], tf.float32, b_init)
                h_1 = tf.nn.tanh(tf.matmul(self.states, w_1) + b_1)
                self.net_params.extend((w_1, b_1))

            with tf.variable_scope('layer_2'):
                w_2 = tf.get_variable(
                    'w', [self.layer_1, self.layer_2], tf.float32, w_init)
                b_2 = tf.get_variable(
                    'b', [self.layer_2], tf.float32, b_init)
                h_2 = tf.nn.tanh(tf.matmul(h_1, w_2) + b_2)
                self.net_params.extend((w_2, b_2))

            with tf.variable_scope('output_layer'):
                w_3 = tf.get_variable(
                    'w', [self.layer_2, self.action_dim], tf.float32, w_init)
                b_3 = tf.get_variable(
                    'b', [self.action_dim], tf.float32, b_init)
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
                h_1 = tf.nn.tanh(
                    tf.matmul(self.target_states, self.target_net_params[0])
                    + self.target_net_params[1])

            with tf.variable_scope('layer_2'):
                h_2 = tf.nn.tanh(
                    tf.matmul(h_1, self.target_net_params[2])
                    + self.target_net_params[3])

            with tf.variable_scope('layer_3'):
                h_3 = tf.nn.tanh(
                    tf.matmul(h_2, self.target_net_params[4])
                    + self.target_net_params[5])
                self.target_output = tf.multiply(h_3, self.action_bound)

    def _build_train_method(self):
        self.action_gradients = tf.placeholder(
            tf.float32, [None, self.action_dim])

        self.actor_gradients = tf.gradients(
            self.output, self.net_params, -self.action_gradients)

        self.optim = tf.train.AdadeltaOptimizer(self.lrate).apply_gradients(
            zip(self.actor_gradients, self.net_params))

    def train(self, action_gradients, batch_state):
        self.sess.run(self.optim, feed_dict={
            self.action_gradients: action_gradients,
            self.states: batch_state
        })

    def update_target_network(self):
        self.sess(self.target_update)

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
