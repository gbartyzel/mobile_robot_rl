import tensorflow as tf

from networks.base import BaseNetwork
from utils.ops import fc_layer, huber_loss


class Critic(BaseNetwork):
    def __init__(self, sess, env):
        super(Critic, self).__init__(sess, env, 'critic')

        self._states,\
        self._actions,\
        self._output,\
        self._net_params = self._build_network('critic_network')

        self._target_states,\
        self._target_actions,\
        self._target_output,\
        self._target_net_params = self._build_network(
            'target_critic_network')

        self._global_step = tf.train.get_or_create_global_step()

        self._build_train_method()
        self._target_update = self._build_update_method()

    @property
    def global_step(self):
        return self._global_step

    def gradients(self, *args):
        return self._sess.run(
            self._action_gradients,
            feed_dict={
                self._states: args[0],
                self._actions: args[1],
            })[0]

    def prediction(self, *args):
        return self._sess.run(
            self._output,
            feed_dict={
                self._target_states: args[0],
                self._target_actions: args[1],
            })

    def target_prediction(self, *args):
        return self._sess.run(
            self._target_output,
            feed_dict={
                self._target_states: args[0],
                self._target_actions: args[1],
            })

    def train(self, *args):
        return self._sess.run(
            [self._optim, self._loss],
            feed_dict={
                self._states: args[0],
                self._actions: args[1],
                self._y: args[2]
            })

    def update_target_network(self, phase='soft_copy'):
        if phase == 'copy':
            self._sess.run([
                self._target_net_params[i].assign(self._net_params[i])
                for i in range(len(self._net_params))
            ])
        else:
            self._sess.run(self._target_update)

    def _build_network(self, name):
        with tf.variable_scope(name):
            with tf.variable_scope('inputs'):
                states = tf.placeholder(tf.float32, [None, self._state_dim],
                                        'input_states')
                actions = tf.placeholder(tf.float32, [None, self._action_dim],
                                         'input_actions')

            h_1 = fc_layer(states, 'layer_1',
                           [self._state_dim, self._layers[0]], tf.nn.relu)

            h_2 = fc_layer(
                tf.concat([h_1, actions], 1), 'layer_2',
                [self._layers[0] + self._action_dim, self._layers[1]],
                tf.nn.relu)

            h_3 = fc_layer(h_2, 'layer_3', [self._layers[1], self._layers[2]],
                           tf.nn.relu)

            output = fc_layer(h_3, 'output_layer', [self._layers[2], 1],
                              tf.identity, 3e-3)

        net_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return states, actions, output, net_params

    def _build_train_method(self):
        with tf.variable_scope('critic_optimizer'):
            self._y = tf.placeholder(tf.float32, [None, 1], 'input_y')

            reg = tf.add_n(
                [self._l2 * tf.nn.l2_loss(var) for var in self._net_params],
                name='l2_reg_term')

            self._loss = tf.add(
                tf.reduce_mean(tf.square(self._y - self._output)),
                reg,
                name='loss')

            self._optim = tf.train.AdamOptimizer(self._lrate).minimize(
                self._loss, global_step=self._global_step)
            self._action_gradients = tf.gradients(self._output, self._actions)

    def _build_update_method(self):
        with tf.variable_scope('critic_to_target'):
            return [
                self._target_net_params[i].assign(
                    tf.multiply(self._target_net_params[i], (1 - self._tau)) +
                    tf.multiply(self._tau, self._net_params[i]))
                for i in range(len(self._net_params))
            ]
