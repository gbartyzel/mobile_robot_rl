import tensorflow as tf

from networks.base import BaseNetwork
from utils.ops import fc_layer, huber_loss


class Critic(BaseNetwork):
    def __init__(self, sess, batch_size):
        super(Critic, self).__init__(sess, 'critic')

        self.states,\
        self.actions,\
        self.q_output,\
        self.net_params = self._build_network('critic_network')

        self.target_states,\
        self.target_actions,\
        self.target_q_output,\
        self.target_net_params = self._build_network(
            'target_critic_network')

        self._build_train_method(batch_size)
        # self._build_summary()
        self.target_update = self._build_update_method()

    def gradients(self, *args):
        return self.sess.run(
            self.action_gradients,
            feed_dict={
                self.states: args[0],
                self.actions: args[1],
            })[0]

    def prediction(self, *args):
        return self.sess.run(
            self.q_output,
            feed_dict={
                self.target_states: args[0],
                self.target_actions: args[1],
            })

    def target_prediction(self, *args):
        return self.sess.run(
            self.target_q_output,
            feed_dict={
                self.target_states: args[0],
                self.target_actions: args[1],
            })

    def train(self, *args):
        return self.sess.run(
            [self.optim, self.loss],
            feed_dict={
                self.states: args[0],
                self.actions: args[1],
                self.y: args[2]
            })

    def update_target_network(self, phase='soft_copy'):
        if phase == 'copy':
            self.sess.run([
                self.target_net_params[i].assign(self.net_params[i])
                for i in range(len(self.net_params))
            ])
        else:
            self.sess.run(self.target_update)

    def _build_network(self, name):
        with tf.variable_scope(name):
            with tf.variable_scope('inputs'):
                states = tf.placeholder(tf.float32, [None, self.state_dim],
                                        'input_states')
                actions = tf.placeholder(tf.float32, [None, self.action_dim],
                                         'input_actions')

            h_1 = fc_layer(states, 'layer_1', [self.state_dim, self.layers[0]],
                           tf.nn.relu)

            h_2 = fc_layer(
                tf.concat([h_1, actions], 1), 'layer_2',
                [self.layers[0] + self.action_dim, self.layers[1]], tf.nn.relu)

            h_3 = fc_layer(h_2, 'layer_3', [self.layers[1], self.layers[2]],
                           tf.nn.relu)

            output = fc_layer(h_3, 'output_layer', [self.layers[2], 1],
                              tf.identity, 3e-3)

        net_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return states, actions, output, net_params

    def _build_train_method(self, batch_size):
        with tf.variable_scope('critic_optimizer'):
            self.y = tf.placeholder(tf.float32, [None, 1], 'input_y')
            """
            reg = tf.add_n(
                [self.l2 * tf.nn.l2_loss(var) for var in self.net_params],
                name='l2_reg_term')

            self.loss = tf.add(
                tf.reduce_mean(tf.square(self.y - self.q_output)),
                reg,
                name='loss')
            """
            self.loss = tf.reduce_mean(
                huber_loss(self.y, self.q_output, 1.0), name='huber_loss')
            self.optim = tf.train.AdamOptimizer(self.lrate).minimize(self.loss)
            grads = tf.gradients(self.q_output, self.actions)
            self.action_gradients = [g / float(batch_size) for g in grads]

    def _build_update_method(self):
        with tf.variable_scope('critic_to_target'):
            return [
                self.target_net_params[i].assign(
                    tf.multiply(self.target_net_params[i], (1 - self.tau)) +
                    tf.multiply(self.tau, self.net_params[i]))
                for i in range(len(self.net_params))
            ]

    def _build_summary(self):
        with tf.variable_scope('critic_summary'):
            for var in self.net_params:
                tf.summary.histogram(var.name, var)

            with tf.name_scope('q_and_loss'):
                self.summary_q = tf.placeholder(tf.float32)
                self.summary_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('q', self.summary_q)
                tf.summary.scalar('loss', self.summary_loss)
