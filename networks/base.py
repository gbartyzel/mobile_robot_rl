class BaseNetwork(object):

    def __init__(self, sess, network_type):
        self.action_dim = 2
        self.state_dim = 7
        self.sess = sess

        if network_type == 'actor':
            self.layers = [400, 300, 300]
            self.lrate = 1e-4

        if network_type == 'critic':
            self.layers = [400, 300, 300]
            self.lrate = 1e-3
            self.l2 = 1e-2

        self.tau = 1e-3

    def prediction(self, *args):
        raise NotImplementedError

    def target_prediction(self, *args):
        raise NotImplementedError

    def train(self, *args):
        raise NotImplementedError

    def update_target_network(self):
        raise NotImplementedError

    def _build_network(self, name):
        raise NotImplementedError

    def _build_train_method(self):
        raise NotImplementedError

    def _build_update_method(self):
        raise NotImplementedError
