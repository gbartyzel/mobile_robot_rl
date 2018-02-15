import tensorflow as tf

from networks.base import BaseNetwork
from utils.ops import batch_norm_layer, fc_layer


class Actor(BaseNetwork):

    def __init__(self, sess, env):
        super(Actor, self).__init__(sess, env, 'actor')

        self.states, \
        self.output, \
        self.is_training, \
        self.net_params = self._build_network('actor_network')

        self.target_states,\
        self.target_output,\
        self.target_is_training,\
        self.target_net_params = self._build_network('target_actor_network')

        self._build_train_method()
        # self._build_summary()
        self.target_update = self._build_update_method()

    def prediction(self, *args):
        return self.sess.run(self.output, feed_dict={
            self.states: args[0],
            self.is_training: False
        })

    def target_prediction(self, *args):
        return self.sess.run(self.target_output, feed_dict={
            self.target_states: args[0],
            self.target_is_training: False
        })

    def train(self, *args):
        self.sess.run(self.optim, feed_dict={
            self.states: args[0],
            self.q_gradients: args[1],
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
                states_bn = batch_norm_layer(states, is_training, tf.identity)

            with tf.variable_scope('bn_layer_1'):
                h_1 = fc_layer(
                    states_bn, 'layer_1', [self.state_dim, self.layers[0]])
                h_1 = batch_norm_layer(h_1, is_training, tf.nn.relu)

            with tf.variable_scope('bn_layer_2'):
                h_2 = fc_layer(
                    h_1, 'layer_2', [self.layers[0], self.layers[1]])
                h_2 = batch_norm_layer(h_2, is_training, tf.nn.relu)

            output = fc_layer(
                h_2, 'output_layer', [self.layers[1], self.action_dim],
                tf.tanh, 3e-3)

        net_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return states, output, is_training, net_params

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
                    tf.multiply(self.target_net_params[i], (1 - self.tau))
                    + tf.multiply(self.tau, self.net_params[i]))
                for i in range(len(self.net_params))
            ]

    def _build_summary(self):
        with tf.variable_scope('actor_layer_1'):
            tf.summary.histogram('w', self.net_params[2])
            tf.summary.histogram('b', self.net_params[3])
        with tf.variable_scope('actor_layer_2'):
            tf.summary.histogram('w', self.net_params[6])
            tf.summary.histogram('b', self.net_params[7])
        with tf.variable_scope('actor_output_layer'):
            tf.summary.histogram('w', self.net_params[10])
            tf.summary.histogram('b', self.net_params[11])

        with tf.variable_scope('target_actor_layer_1'):
            tf.summary.histogram('w', self.target_net_params[2])
            tf.summary.histogram('b', self.target_net_params[3])
        with tf.variable_scope('target_actor_layer_2'):
            tf.summary.histogram('w', self.target_net_params[6])
            tf.summary.histogram('b', self.target_net_params[7])
        with tf.variable_scope('target_actor_output_layer'):
            tf.summary.histogram('w', self.target_net_params[10])
            tf.summary.histogram('b', self.target_net_params[11])
