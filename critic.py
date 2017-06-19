import tensorflow as tf

class Critic(object):

    def __init__(self, sess, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.sess = sess

        self.initializer = tf.random_normal_initializer(stddev=0.01)
    def _build_network(self):
        self.net_params = []
        with tf.variable_scope('critic_network'):
            self.states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'input_states')

            with tf.variable_scope('layer_1'):
                w_1 = tf.get_variable(
                    'w', [self.state_dim, 400], tf.float32, self.initializer)
                b_1 = tf.get_variable(
                    'b', [400], tf.float32, tf.zeros_initializer(0.0))
                h_1 = tf.nn.tanh(tf.matmul(self.states, w_1) + b_1)
                self.net_params.extend((w_1, b_1))

            with tf.variable_scope('layer_2'):
                