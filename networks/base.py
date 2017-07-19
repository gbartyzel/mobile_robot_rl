import numpy as np
import tensorflow as tf

SEED = 1337


class BaseNetwork(object):

    def __init__(self, config, network_type):
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim

        if network_type == 'actor':
            self.layer_1 = config.alayer_1
            self.layer_2 = config.alayer_2
            self.action_bound = config.action_bound
            self.lrate = config.alearning_rate

        if network_type == 'critic':
            self.layer_1 = config.clayer_1
            self.layer_2 = config.clayer_2
            self.lrate = config.clearning_rate

        self.seed = config.seed
        self.tau = config.tau

        self.weight_decay = tf.constant(0.01, tf.float32, name='weight_decay')

    def save(self, time_stamp):
        pass

    def _build_network(self, name):
        """
        Overwrite this method by method which creates basic network
        of the model.
        """
        pass

    def _build_train_method(self):
        """
        Overwrite this method by training method of the model's network.
        """
        pass

    def _build_saver(self, name, net, target_net):
        with tf.variable_scope(name):
            saver = tf.train.Saver(net, name='saver')
            target_saver = tf.train.Saver(target_net, name='target_saver')
        return saver, target_saver

    def _variable(self, name, shape, fan, add_reg=False):
        val = 1 / np.sqrt(fan)
        init = tf.random_uniform_initializer(-val, val, dtype=tf.float32)

        if add_reg:
            reg = tf.contrib.layers.l2_regularizer(self.weight_decay)
        else:
            reg = None

        return tf.get_variable(name, shape, tf.float32, init, reg)

    def _batch_norm_layer(self, x, train_phase, activation=None):
        return tf.contrib.layers.batch_norm(
                x,
                activation_fn=activation,
                center=True,
                scale=True,
                updates_collections=None,
                is_training=train_phase,
                reuse=None,
                decay=0.9,
                epsilon=1e-5)
