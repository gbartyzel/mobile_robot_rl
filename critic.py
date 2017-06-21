import tensorflow as tf


class Critic(object):

    def __init__(self, sess, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.sess = sess

        self._build_network()
        self._build_target_network()
        self._build_train_method()

        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

    def _build_network(self):
        initializer = tf.random_normal_initializer(stddev=0.01)

        self.net_params = []
        with tf.variable_scope('critic_network'):
            self.states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'input_states')
            self.actions = tf.placeholder(
                tf.float32, [None, self.action_dim], 'input_actions')

            with tf.variable_scope('layer_1'):
                w_1 = tf.get_variable(
                    'w', [self.state_dim, 400], tf.float32, initializer)
                b_1 = tf.get_variable(
                    'b', [400], tf.float32, tf.zeros_initializer(0.0))
                h_1 = tf.nn.tanh(tf.matmul(self.states, w_1) + b_1)
                self.net_params.extend((w_1, b_1))

            with tf.variable_scope('layer_2'):
                w_2 = tf.get_variable(
                    'w', [400, 300], tf.float32, initializer)
                w_2_a = tf.get_variable(
                    'w', [self.action_dim, 300], tf.float32, initializer)
                b_2 = tf.get_variable(
                    'b', [300], tf.float32, initializer)
                h_2 = tf.nn.tanh(tf.matmul(h_1, w_2)
                                 + tf.matmul(self.actions, w_2_a) + b_2)
                self.net_params.extend((w_2, w_2_a, b_2))

            with tf.variable_scope('output_layer'):
                w_3 = tf.get_variable(
                    'w', [300, 1], tf.float32, initializer)
                b_3 = tf.get_variable(
                    'b', [1], tf.zeros_initializer(0.0))
                self.q_output = tf.matmul(h_2, w_3) + b_3

    def _build_target_network(self):
        with tf.variable_scope('actor_target_model'):
            self.target_states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'target_states')
            self.target_actions = tf.placeholder(
                tf.float32, [None, self.action_dim], 'target_states')

            ema = tf.train.ExponentialMovingAverage(decay=(1.-0.001))
            self.target_update = ema.apply(self.net_params)
            self.target_net_params = [ema.average(x) for x in self.net_params]

            with tf.variable_scope('layer_1'):
                h_1 = tf.nn.tanh(
                    tf.matmul(self.target_states, self.target_net_params[0])
                    + self.target_net_params[1])

            with tf.variable_scope('layer_2'):
                h_2 = tf.nn.tanh(
                    tf.matmul(h_1, self.target_net_params[2])
                    + tf.matmul(self.target_actions, self.target_net_params[3])
                    + self.target_net_params[4])

            with tf.variable_scope('layer_3'):
                self.target_q_output = (
                    tf.matmul(h_2, self.target_net_params[5])
                    + self.target_net_params[6])

    def _build_train_method(self):
        self.y = tf.placeholder(tf.float32, [None], 1)
        weight_decay = tf.add_n(
            [0.01 * tf.nn.l2_loss(var) for var in self.net_params])
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.y, self.q_output)) + weight_decay
        self.optim = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.gradients = tf.gradients(self.q_output, self.actions)

    def train(self, batch_y, batch_state, batch_action):
        self.sess.run(self.optim, feed_dict={
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
        return self.sess.run(self.gradients, feed_dict={
            self.states: batch_state,
            self.actions: batch_action
        })

    def update_target_network(self):
        self.sess.run(self.target_update)
