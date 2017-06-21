import tensorflow as tf


class Actor(object):
    
    def __init__(self, sess, action_dim, state_dim, action_bound):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.sess = sess

        self._build_network()
        self._build_target_network()
        self._build_train_method()

        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

    def _build_network(self):
        initializer = tf.random_normal_initializer(stddev=0.01)
        self.net_params = []
        with tf.variable_scope('actor_model'):
            self.states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'states')

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
                b_2 = tf.get_variable(
                    'b', [300], tf.float32, tf.zeros_initializer(0.0))
                h_2 = tf.nn.tanh(tf.matmul(h_1, w_2) + b_2)
                self.net_params.extend((w_2, b_2))

            with tf.variable_scope('output_layer'):
                w_3 = tf.get_variable(
                    'w', [300, self.action_dim], tf.float32, initializer)
                b_3 = tf.get_variable(
                    'b', [self.action_dim], tf.float32,
                    tf.zeros_initializer(0.0))
                h_3 = tf.nn.tanh(tf.matmul(h_2, w_3) + b_3)
                self.net_params.extend((w_3, b_3))
                self.output = tf.multiply(h_3, self.action_bound)

    def _build_target_network(self):
        with tf.variable_scope('actor_target_model'):
            self.target_states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'target_states')
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

        self.optim = tf.train.AdadeltaOptimizer(0.00025).apply_gradients(
            zip(self.actor_gradients, self.net_params))

    def train(self, action_gradients, batch_state, ):
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
