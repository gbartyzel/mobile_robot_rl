import gym


class NavigationWrapper(gym.Wrapper):
    def __init__(self, env):
        super(NavigationWrapper, self).__init__(env)
        self._u_min = -1.0
        self._u_max = 1.0

    def step(self, action):
        t_max = self.env.action_space.high
        t_min = self.env.action_space.low
        new_action = (action - self._u_min) / (self._u_max - self._u_min) \
                     * (t_max - t_min) + t_min
        return self.env.step(new_action)
