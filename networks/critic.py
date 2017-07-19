import tensorflow as tf
from networks.base import BaseNetwork


class Critic(BaseNetwork):

    def __init__(self, sess, config):
        super(Critic, self).__init__(config, 'critic')
        self.sess = sess

        self.states,\
        self.actions,\
        self.q_output,\
        self.net_params = self._build_network('critic_network')

        self.target_states,\
        self.target_actions,\
        self.target_update,\
        self.target_q_output,\
        self.target_net_params = self._build_target_network(
            'target_critic_network', self.net_params)

        self._build_train_method()

        self._build_summary()

        self.saver, self.target_saver = self._build_saver(
            'critic_saver', self.net_params, self.target_net_params)

    def train(self, batch_y, batch_state, batch_action):
        return self.sess.run([self.optim, self.loss], feed_dict={
            self.y: batch_y,
            self.states: batch_state,
            self.actions: batch_action,
        })

    def update_target_network(self, phase='run'):
        """
        if phase == 'first_run':
            self.sess.run([
                self.target_net_params[i].assign(self.net_params[i])
                for i in range(len(self.net_params))
            ])
        """
        self.sess.run(self.target_update)

    def predict(self, batch_action, batch_state):
        return self.sess.run(self.q_output, feed_dict={
            self.states: batch_state,
            self.actions: batch_action,
            })

    def target_prediction(self, batch_action, batch_state):
        return self.sess.run(self.target_q_output, feed_dict={
            self.target_states: batch_state,
            self.target_actions: batch_action,
        })

    def gradients(self, batch_state, batch_action):
        return self.sess.run(self.action_gradients, feed_dict={
            self.states: batch_state,
            self.actions: batch_action,
        })[0]

    def save(self, time_stamp):
        self.saver.save(self.sess,
                        'saved_networks/critic/model',
                        global_step=time_stamp)
        self.target_saver.save(self.sess,
                               'saved_networks/target_critic/model',
                               global_step=time_stamp)

    def _build_network(self, name):
        with tf.variable_scope(name):
            states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'input_states')
            actions = tf.placeholder(
                tf.float32, [None, self.action_dim], 'input_actions')

            with tf.variable_scope('layer_1'):
                w_1 = self._variable(
                    'w', [self.state_dim, self.layer_1], self.state_dim, True)
                b_1 = self._variable('b', [self.layer_1], self.state_dim, True)
                h_1 = tf.nn.tanh(tf.matmul(states, w_1) + b_1)

            with tf.variable_scope('layer_2'):
                w_2 = self._variable(
                    'w', [self.layer_1, self.layer_2],
                    self.layer_1+self.action_dim, True)
                w_2_a = self._variable(
                    'w_a', [self.action_dim, self.layer_2],
                    self.layer_1 + self.action_dim, True)
                b_2 = self._variable(
                    'b', [self.layer_2], self.layer_1 + self.action_dim)
                h_2 = tf.nn.tanh(
                    tf.matmul(h_1, w_2) + tf.matmul(actions, w_2_a) + b_2)

            with tf.variable_scope('output_layer'):
                w_3 = tf.get_variable(
                    'w', [self.layer_2, 1], tf.float32,
                    tf.random_uniform_initializer(-3e-4, 3e-4),
                    tf.contrib.layers.l2_regularizer(self.weight_decay))
                b_3 = tf.get_variable(
                    'b', [1], tf.float32,
                    tf.random_uniform_initializer(-3e-3, 3e-3))
                q_output = tf.identity(tf.matmul(h_2, w_3) + b_3)

        return states, actions, q_output, [w_1, b_1, w_2, w_2_a, b_2, w_3, b_3]

    def _build_target_network(self, name, net):
        with tf.variable_scope(name):
            states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'input_states')
            actions = tf.placeholder(
                tf.float32, [None, self.action_dim], 'input_actions')

            ema = tf.train.ExponentialMovingAverage(decay=1-self.tau)
            target_update = ema.apply(net)
            target_net = [ema.average(x) for x in net]

            with tf.variable_scope('layer_1'):
                h_1 = tf.nn.tanh(
                    tf.matmul(states, target_net[0]) + target_net[1])

            with tf.variable_scope('layer_2'):
                h_2 = tf.nn.tanh(
                    tf.matmul(h_1, target_net[2])
                    + tf.matmul(actions, target_net[3])
                    + target_net[4])

            with tf.variable_scope('output_layer'):
                output = tf.tanh(
                    tf.matmul(h_2, target_net[5]) + target_net[6])

            return states, actions, target_update, output, target_net

    def _build_train_method(self):
        with tf.variable_scope('critic_optimizer'):
            self.y = tf.placeholder(tf.float32, [None, 1], 'input_y')

            reg_variables = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, 'critic_network')
            reg_term = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(self.weight_decay),
                reg_variables)

            self.loss = tf.add(
                tf.reduce_mean(tf.square(self.y - self.q_output), name='loss'),
                reg_term)

            self.optim = tf.train.AdamOptimizer(self.lrate).minimize(self.loss)
            self.action_gradients = tf.gradients(self.q_output, self.actions)

    def _build_update_method(self):
        with tf.variable_scope('critic_to_target'):
            return [
                self.target_net_params[i].assign(
                    tf.add(tf.multiply(self.target_net_params[i],
                                       (1 - self.tau)),
                           tf.multiply(self.tau, self.net_params[i])))
                for i in range(len(self.net_params))
            ]

    def _build_summary(self):
        with tf.variable_scope('critic_summary'):
            with tf.variable_scope('layer_1'):
                tf.summary.histogram('w', self.net_params[0])
                tf.summary.histogram('b', self.net_params[1])
            with tf.variable_scope('layer_2'):
                tf.summary.histogram('w', self.net_params[2])
                tf.summary.histogram('w_a', self.net_params[3])
                tf.summary.histogram('b', self.net_params[4])
            with tf.variable_scope('output_layer'):
                tf.summary.histogram('w', self.net_params[5])
                tf.summary.histogram('b', self.net_params[6])

        with tf.variable_scope('target_critic_summary'):
            with tf.variable_scope('layer_1'):
                tf.summary.histogram('w_t', self.target_net_params[0])
                tf.summary.histogram('b_t', self.target_net_params[1])
            with tf.variable_scope('layer_2'):
                tf.summary.histogram('w_t', self.target_net_params[2])
                tf.summary.histogram('w_ta', self.target_net_params[3])
                tf.summary.histogram('b_t', self.target_net_params[4])
            with tf.variable_scope('output_layer'):
                tf.summary.histogram('w_t', self.target_net_params[5])
                tf.summary.histogram('b_t', self.target_net_params[6])
