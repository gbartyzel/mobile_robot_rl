import os
import time

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
        self.actor = Actor(self.sess, self.config)
        self.critic = Critic(self.sess, self.config)
        self.ou_noise = OUNoise(self.config)
        self.memory = ReplayMemory(self.config.memory_size)

    def learn(self):
        train_batch = self.memory.sample(self.config.batch_size)

        next_action = self.actor.target_actions(train_batch[:,0])
        q_value = self.critic.predict(train_batch[:,1], train_batch[:,0])

        done = train_batch[:,4] + 0
        y_batch = (1. - done) * self.config.gamma * q_value + train_batch[:,1]
        y_batch = np.resize(y_batch,[self.config.batch_size,1])

        self.critic.train(y_batch, train_batch[:,0], train_batch[:,1])

        grad_actions = self.actor.actions(train_batch[:,0]) 
        gradients = self.critic.gradients(train_batch[:,0], grad_actions)
        self.actor.train(grad_actions, train_batch[:,0])

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
        return self.actor.action(state) + self.ou_noise.noise()

    def action(self, state):
        return self.actor.action(state)
