import logging
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from environment.env import Env
from networks.actor import Actor
from networks.critic import Critic
from utils.memory import ReplayMemory
from utils.ounoise import OUNoise
from vrep import vrep


class DDPGAgent(object):

    def __init__(self, sess, goal, config):
        self.config = config
        self.goal = goal
        self.sess = sess

        self.epsilon = 1.0

        self._build_summary()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("output", sess.graph)

        self.critic = Critic(self.sess, self.config)
        self.actor = Actor(self.sess, self.config)
        self.ou_noise = OUNoise(self.config)
        self.memory = ReplayMemory(self.config.memory_size)

    def train(self):

        logging.basicConfig(filename='learn_log.log', level=logging.INFO)
        for ep in tqdm(range(self.config.num_episode)):
            start_time = time.time()
            vrep.simxFinish(-1)
            env = Env(self.config, self.goal, 'train')
            client = env.client
            if client != -1:
                print("Connected to V-REP server")
                state = env.state
                vrep.simxSynchronousTrigger(client)
                done = False
                step = 0
                while step < self.config.sim_time:
                    if not done:
                        action = self.noise_action(state)
                        reward, next_state, done = env.step(action)
                        self.observe(state, action, reward, next_state, done)
                        state = next_state
                        step += 1
                        self.total_steps += 1
                        if self.total_steps % 1000 == 0:
                            self.save(int(self.total_steps))
                    else:
                        vrep.simxStopSimulation(client,
                                                vrep.simx_opmode_oneshot)
                        if vrep.simxGetConnectionId(client) == -1:
                            break
            else:
                print("Couldn't connect to V-REP server!")
            vrep.simxFinish(client)
            elapsed = time.time() - start_time
            elapsed_time = str(timedelta(seconds=int(elapsed)))
            logging.info('Time elapsed: %s' % elapsed_time)
            if ep % self.config.test_step == self.config.test_step - 1:
                self.play(ep)

    def train_mini_batch(self):
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
        gradient_actions = self.actor.actions(state_batch)
        q_gradients = self.critic.gradients(state_batch, gradient_actions)
        self.actor.train(q_gradients, state_batch)

        self.actor.update_target_network()
        self.critic.update_target_network()

    def observe(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if self.memory.size >= self.config.start_learning:
            self.train_mini_batch()

        if done:
            self.ou_noise.reset()

    def play(self, ep):
        max_reward = min_reward = avg_reward = avg_steps = 0.0
        path = np.zeros(self.config.test_trial)
        nav_error = np.zeros(self.config.test_trial)
        for i in range(self.config.test_trial):
            vrep.simxFinish(-1)
            env = Env(self.config, self.goal, 'test')
            client = env.client
            if client != -1:
                print("Connected to V-REP server")
                state = env.state
                vrep.simxSynchronousTrigger(client)
                done = False
                step = 0
                ep_reward = 0.0
                while step < self.config.sim_time:
                    if not done:
                        action = self.action(state)
                        reward, next_state, done = env.step(action)
                        state = next_state
                        ep_reward += reward
                        step += 1
                        path[i] = env.path
                        nav_error[i] = env.current_error
                    else:
                        vrep.simxStopSimulation(client,
                                                vrep.simx_opmode_oneshot)
                        if vrep.simxGetConnectionId(client) == -1:
                            break
                avg_reward += ep_reward
                avg_steps += step
                if i == 0:
                    max_reward = min_reward = ep_reward
                else:
                    if ep_reward > max_reward:
                        max_reward = ep_reward
                    if ep_reward < min_reward:
                        min_reward = ep_reward
            else:
                print("Couldn't connect to V-REP server!")
            vrep.simxFinish(client)
        avg_steps /= self.config.test_trial
        avg_reward /= self.config.test_trial
        avg_path = np.mean(path)
        avg_error = np.mean(nav_error)
        summary = self.sess.run(self.merged, feed_dict={
            self.avg_reward: avg_reward,
            self.min_reward: min_reward,
            self.max_reward: max_reward,
            self.avg_steps: avg_steps,
            self.avg_distance: avg_path,
            self.avg_error: avg_error
        })
        self.writer.add_summary(summary, int(ep))

    def noise_action(self, state):
        factor = 1/self.config.explore
        self.epsilon -= factor
        self.epsilon = max(self.epsilon, 0)
        noise = self.epsilon * self.ou_noise.noise()

        return np.clip(self.actor.action(state) + noise, 0.0, 1.0)

    def action(self, state):
        return self.actor.action(state)

    def buffer_size(self):
        return self.memory.size

    def save(self, time_stamp):
        self.actor.save(time_stamp)
        self.critic.save(time_stamp)

    def _build_summary(self):
        self.total_steps = 0.0
        with tf.variable_scope('summary'):
            self.avg_reward = tf.placeholder(tf.float32)
            self.min_reward = tf.placeholder(tf.float32)
            self.max_reward = tf.placeholder(tf.float32)
            self.avg_steps = tf.placeholder(tf.float32)
            self.avg_error = tf.placeholder(tf.float32)
            self.avg_distance = tf.placeholder(tf.float32)
            tf.summary.scalar('avg_reward', self.avg_reward)
            tf.summary.scalar('max_reward', self.max_reward)
            tf.summary.scalar('min_reward', self.min_reward)
            tf.summary.scalar('avg_steps', self.avg_steps)
            tf.summary.scalar('avg_error', self.avg_error)
            tf.summary.scalar('avg_distance', self.avg_distance)
