from networks.config import MODEL_CONFIG


class BaseNetwork(object):
    def __init__(self, sess, env, network_type):
        self._action_dim = env.action_dim
        self._state_dim = env.observation_dim
        self._sess = sess

        if network_type == 'actor':
            self._layers = MODEL_CONFIG['actor']['layers']
            self._lrate = MODEL_CONFIG['actor']['learning_rate']

        if network_type == 'critic':
            self._layers = MODEL_CONFIG['critic']['layers']
            self._lrate = MODEL_CONFIG['critic']['learning_rate']
            self._l2 = MODEL_CONFIG['critic']['l2_rate']

        self._tau = MODEL_CONFIG['tau']

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
