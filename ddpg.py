import numpy as np
import tensorflow as tf

from networks.actor import Actor
from networks.critic import Critic
from utils.memory import ReplayMemory
from utils.ounoise import OUNoise


class DDPGAgent(object):

    def __init__(self, sess, goal, config):
        self.config = config
        self.goal = goal
        self.sess = sess

        self.critic = Critic(self.sess, self.config)
        self.actor = Actor(self.sess, self.config)
        self.ou_noise = OUNoise(self.config)
        self.memory = ReplayMemory(self.config.memory_size)

    def learn(self):
        train_batch = self.memory.sample(self.config.batch_size)
        state_batch = np.vstack(train_batch[:, 0])
        action_batch = np.vstack(train_batch[:, 1])
        reward_batch = train_batch[:, 2]
        next_state_batch = np.vstack(train_batch[:, 3])
        done_batch = train_batch[:, 4]

        next_action = self.actor.target_actions(next_state_batch)
        q_value = self.critic.target_prediction(next_action, next_state_batch)
        done = done_batch + 0.0
        y_batch = (1. - done) * self.config.gamma * q_value + reward_batch
        y_batch = np.resize(y_batch, [self.config.batch_size, 1])

        _, loss = self.critic.train(y_batch, state_batch, action_batch)
        print(loss)
        gradient_actions = self.actor.actions(state_batch)
        q_gradients = self.critic.gradients(state_batch, gradient_actions)
        self.actor.train(q_gradients, state_batch)

        self.actor.update_target_network()
        self.critic.update_target_network()

    def observe(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if self.memory.size >= self.config.start_learning:
            self.learn()

        if done:
            self.ou_noise.reset()

    def test(self):
        pass

    def noise_action(self, state):
        return np.clip(self.actor.action(state) + self.ou_noise.noise(),
                       -1.0, 1.0)

    def action(self, state):
        return self.actor.action(state)

    def buffer_size(self):
        return self.memory.size

    def save(self, time_stamp):
        self.actor.save(time_stamp)
        self.critic.save(time_stamp)

    def _build_summary(self):
        pass
