import tensorflow as tf

from networks.base import BaseNetwork
from utils.ops import batch_norm_layer, fc_layer, clipped_error


class Critic(BaseNetwork):

    def __init__(self, sess, env):
        super(Critic, self).__init__(sess, env, 'critic')

        self.states,\
        self.actions,\
        self.q_output,\
        self.is_training,\
        self.net_params = self._build_network('critic_network')

        self.target_states,\
        self.target_actions,\
        self.target_q_output,\
        self.target_is_training,\
        self.target_net_params = self._build_network('target_critic_network')

        self._build_train_method()
        # self._build_summary()
        self.target_update = self._build_update_method()

    def gradients(self, *args):
        return self.sess.run(self.action_gradients, feed_dict={
            self.states: args[0],
            self.actions: args[1],
            self.is_training: False
        })[0]

    def prediction(self, *args):
        return self.sess.run(self.q_output, feed_dict={
            self.target_states: args[0],
            self.target_actions: args[1],
            self.is_training: False
        })

    def target_prediction(self, *args):
        return self.sess.run(self.target_q_output, feed_dict={
            self.target_states: args[0],
            self.target_actions: args[1],
            self.target_is_training: False
        })

    def train(self, *args):
        return self.sess.run([self.optim, self.loss], feed_dict={
            self.states: args[0],
            self.actions: args[1],
            self.y: args[2],
            self.is_training: True
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
                is_training = tf.placeholder(tf.bool, name='is_training')
                states = tf.placeholder(
                    tf.float32, [None, self.state_dim], 'input_states')
                actions = tf.placeholder(
                    tf.float32, [None, self.action_dim], 'input_actions')
                states_bn = batch_norm_layer(states, is_training, tf.identity)
                actions_bn = batch_norm_layer(actions, is_training, tf.identity)

            with tf.variable_scope('bn_layer_1'):
                h_1 = fc_layer(
                    states_bn, 'layer_1', [self.state_dim, self.layers[0]],
                    xavier=True)
                h_1 = batch_norm_layer(h_1, is_training, tf.nn.relu)

            with tf.variable_scope('bn_layer_2'):
                h_2 = fc_layer(
                    tf.concat([h_1, actions_bn], 1), 'layer_2',
                    [self.layers[0] + self.action_dim, self.layers[1]],
                    xavier=True)
                h_2 = batch_norm_layer(h_2, is_training, tf.nn.relu)

            output = fc_layer(
                h_2, 'output_layer', [self.layers[1], 1],
                tf.identity, 3e-4)

        net_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return states, actions, output, is_training, net_params

    def _build_train_method(self):
        with tf.variable_scope('critic_optimizer'):
            self.y = tf.placeholder(tf.float32, [None, 1], 'input_y')
            """
            reg = tf.add_n([
                self.l2 * tf.nn.l2_loss(var) for var in self.net_params
                if 'weight' in var.name
                ], name='l2_reg_term')

            self.loss = tf.add(
                tf.reduce_mean(tf.square(self.y - self.q_output)),
                reg, name='loss')
            """
            self.loss = tf.reduce_mean(clipped_error(self.y - self.q_output))
            self.optim = tf.train.AdamOptimizer(self.lrate).minimize(self.loss)
            grads = tf.gradients(self.q_output, self.actions)
            self.action_gradients = [g / float(64) for g in grads]

    def _build_update_method(self):
        with tf.variable_scope('critic_to_target'):
            return [
                self.target_net_params[i].assign(
                    tf.multiply(self.target_net_params[i], (1 - self.tau))
                    + tf.multiply(self.tau, self.net_params[i]))
                for i in range(len(self.net_params))
            ]

    def _build_summary(self):
        with tf.name_scope('critic_layer_1'):
            tf.summary.histogram('w', self.net_params[4])
            tf.summary.histogram('b', self.net_params[5])
        with tf.name_scope('critic_layer_2'):
            tf.summary.histogram('w', self.net_params[8])
            tf.summary.histogram('b', self.net_params[9])
        with tf.name_scope('critic_layer_3'):
            tf.summary.histogram('w', self.net_params[12])
            tf.summary.histogram('b', self.net_params[13])

        with tf.name_scope('target_critic_layer_1'):
            tf.summary.histogram('w', self.target_net_params[4])
            tf.summary.histogram('b', self.target_net_params[5])
        with tf.name_scope('target_critic_layer_2'):
            tf.summary.histogram('w', self.target_net_params[8])
            tf.summary.histogram('b', self.target_net_params[9])
        with tf.name_scope('target_critic_layer_3'):
            tf.summary.histogram('w', self.target_net_params[12])
            tf.summary.histogram('b', self.target_net_params[13])

        with tf.name_scope('q_and_loss'):
            self.summary_q = tf.placeholder(tf.float32)
            tf.summary.scalar('q', self.summary_q)
