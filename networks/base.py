class BaseNetwork(object):

    def __init__(self, config, network_type):
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim

        if network_type == 'actor':
            self.layer_1 = config.alayer_1
            self.layer_2 = config.alayer_2
            self.action_bound = self.action_bound

        if network_type == 'critic':
            self.layer_1 = config.clayer_1
            self.layer_2 = config.clayer_2

        self.lrate = config.learning_rate
        self.tau = config.tau

    def _build_network(self):
        '''
        Overwrite this method by method which creates basic network
        of the model.
        '''
        pass

    def _build_target_network(self):
        '''
        Overwrite this method by method which creates target network
        of the model.
        '''
        pass

    def _build_train_method(self):
        '''
        Overwrite this method by training method of the model's network.
        '''
        pass
