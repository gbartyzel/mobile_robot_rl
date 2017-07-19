import tensorflow as tf

from networks.base import BaseNetwork


class Actor(BaseNetwork):

    def __init__(self, sess, config):
        super(Actor, self).__init__(config, 'actor')
        self.sess = sess

        self.states, \
        self.is_training,\
        self.output, \
        self.net_params = self._build_network('actor_network')

        self.target_states,\
        self.target_is_training,\
        self.target_update,\
        self.target_output,\
        self.target_net_params = self._build_target_network(
            'target_actor_network', self.net_params)

        self._build_train_method()

        # self.target_update = self._build_update_method()

        self._build_summary()

        self.saver, self.target_saver = self._build_saver(
            'actor_saver', self.net_params, self.target_net_params)

    def train(self, action_gradients, batch_state):
        self.sess.run(self.optim, feed_dict={
            self.q_gradients: action_gradients,
            self.states: batch_state,
            self.is_training: True
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

    def load(self):
        checkpoint = tf.train.get_checkpoint_state("saved_networks/actor")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save(self, time_stamp):
        self.saver.save(self.sess,
                        'saved_networks/actor/model',
                        global_step=time_stamp)
        self.target_saver.save(self.sess,
                               'saved_networks/target_actor/model',
                               global_step=time_stamp)

    def _build_network(self, name):
        with tf.variable_scope(name):
            is_training = tf.placeholder(tf.bool, name='is_training')
            states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'input_states')
            states_bn = self._batch_norm_layer(states, is_training, tf.identity)

            with tf.variable_scope('layer_1'):
                w_1 = self._variable(
                    'w', [self.state_dim, self.layer_1], self.state_dim, True)
                b_1 = self._variable('b', [self.layer_1], self.state_dim)
                h_1 = tf.matmul(states_bn, w_1) + b_1
                h_1_bn = self._batch_norm_layer(h_1, is_training, tf.nn.tanh)

            with tf.variable_scope('layer_2'):
                w_2 = self._variable(
                    'w', [self.layer_1, self.layer_2], self.layer_1, True)
                b_2 = self._variable('b', [self.layer_2], self.layer_1)
                h_2 = tf.matmul(h_1_bn, w_2) + b_2
                h_2_bn = self._batch_norm_layer(h_2, is_training, tf.nn.tanh)

            with tf.variable_scope('output_layer'):
                w_3 = tf.get_variable(
                    'w', [self.layer_2, self.action_dim], tf.float32,
                    tf.random_uniform_initializer(-3e-3, 3e-3),
                    tf.contrib.layers.l2_regularizer(self.weight_decay))
                b_3 = tf.get_variable(
                    'b', [self.action_dim], tf.float32,
                    tf.random_uniform_initializer(-3e-3, 3e-3))
                output = tf.tanh(tf.matmul(h_2_bn, w_3) + b_3)

        return states, is_training, output, [w_1, b_1, w_2, b_2, w_3, b_3]

    def _build_target_network(self, name, net):
        with tf.variable_scope(name):
            states = tf.placeholder(
                tf.float32, [None, self.state_dim], 'input_states')
            is_training = tf.placeholder(tf.bool, name='is_training')
            states_bn = self._batch_norm_layer(states, is_training, tf.identity)

            ema = tf.train.ExponentialMovingAverage(decay=1-self.tau)
            target_update = ema.apply(net)
            target_net = [ema.average(x) for x in net]

            with tf.variable_scope('layer_1'):
                h_1 = tf.matmul(states_bn, target_net[0]) + target_net[1]
                h_1_bn = self._batch_norm_layer(h_1, is_training, tf.nn.tanh)

            with tf.variable_scope('layer_2'):
                h_2 = tf.matmul(h_1_bn, target_net[2]) + target_net[3]
                h_2_bn = self._batch_norm_layer(h_2, is_training, tf.nn.tanh)

            with tf.variable_scope('output_layer'):
                output = tf.tanh(
                    tf.matmul(h_2_bn, target_net[4]) + target_net[5])

            return states, is_training, target_update, output, target_net

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

    def _build_summary(self):
        with tf.variable_scope('actor_summary'):
            with tf.variable_scope('layer_1'):
                tf.summary.histogram('w1', self.net_params[0])
                tf.summary.histogram('b1', self.net_params[1])
            with tf.variable_scope('layer_2'):
                tf.summary.histogram('w2', self.net_params[2])
                tf.summary.histogram('b2', self.net_params[3])
            with tf.variable_scope('output_layer'):
                tf.summary.histogram('w3', self.net_params[4])
                tf.summary.histogram('b3', self.net_params[5])

        with tf.variable_scope('target_critic_summary'):
            with tf.variable_scope('layer_1'):
                tf.summary.histogram('w_t', self.target_net_params[0])
                tf.summary.histogram('b_t', self.target_net_params[1])
            with tf.variable_scope('layer_2'):
                tf.summary.histogram('w_t', self.target_net_params[2])
                tf.summary.histogram('b_t', self.target_net_params[3])
            with tf.variable_scope('output_layer'):
                tf.summary.histogram('w_t', self.target_net_params[4])
                tf.summary.histogram('b_t', self.target_net_params[5])
