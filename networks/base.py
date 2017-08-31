import tensorflow as tf


class BaseNetwork(object):

    def __init__(self, config, network_type, sess):
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        self.sess = sess

        if network_type == 'actor':
            self.layers = config.actor_layers
            self.action_bound = config.action_bound
            self.lrate = config.alearning_rate

        if network_type == 'critic':
            self.layers = config.critic_layers
            self.lrate = config.clearning_rate

        self.tau = config.tau

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

    def load(self, network):
        saver = tf.train.Saver(name=network)
        path = 'saved_' + network + "/model"
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")


