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

    def __init__(self, goal, config):
        self.config = config
        self.goal = goal
        self.sess = tf.InteractiveSession()

        self.total_steps = 0.0
        self._build_summary()

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("train_output", self.sess.graph)

        self.critic = Critic(self.sess, self.config)
        self.actor = Actor(self.sess, self.config)
        self.ou_noise = OUNoise(self.config)
        self.memory = ReplayMemory(self.config.memory_size)

    def train(self):
        for ep in tqdm(range(self.config.num_episode)):
            env = Env(self.goal)
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
                        action = self.noise_action(state)
                        reward, next_state, done = env.step(action)
                        self.observe(state, action, reward, next_state, done)
                        state = next_state
                        step += 1
                        ep_reward += reward
                        self.total_steps += 1
                    else:
                        vrep.simxStopSimulation(
                            client, vrep.simx_opmode_oneshot)
                        if vrep.simxGetConnectionId(client) == -1:
                            break
                print("Reward: ", ep_reward, "Steps: ", step)
            else:
                print("Couldn't connect to V-REP server!")
            if ep % self.config.test_step == self.config.test_step - 1:
                result = self.test()
                summary = self.sess.run(self.merged, feed_dict={
                    self.avg_reward: np.mean(result[0]),
                    self.min_reward: np.min(result[0]),
                    self.max_reward: np.max(result[0]),
                    self.avg_steps: np.mean(result[1]),
                    self.avg_distance: np.mean(result[2]),
                    self.avg_error: np.mean(result[3])
                })
                self.writer.add_summary(summary, int(ep))

    def train_mini_batch(self):
        train_batch = self.memory.sample(self.config.batch_size)
        state_batch = np.vstack(train_batch[:, 0])
        action_batch = np.vstack(train_batch[:, 1])
        reward_batch = train_batch[:, 2]
        next_state_batch = np.vstack(train_batch[:, 3])
        done_batch = train_batch[:, 4]

        next_action = self.actor.target_actions(next_state_batch)
        q_value = self.critic.target_prediction(next_action, next_state_batch)

        done_batch += 0.0
        y_batch = (1. - done_batch) * self.config.gamma * q_value + reward_batch
        y_batch = np.resize(y_batch, [self.config.batch_size, 1])
        self.critic.train(y_batch, state_batch, action_batch)

        gradient_actions = self.actor.actions(state_batch)
        q_gradients = self.critic.gradients(state_batch, gradient_actions)
        self.actor.train(q_gradients, state_batch)

        self.actor.update_target_network()
        self.critic.update_target_network()

    def observe(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if self.memory.count() >= self.config.start_learning:
            self.train_mini_batch()

        if self.total_steps % 10000 == 0:
            self.actor.save(self.total_steps)
            self.critic.save(self.total_steps)

        if done:
            self.ou_noise.reset()

    def test(self):
        rewards = np.zeros(self.config.test_trial)
        steps = np.zeros(self.config.test_trial)
        path = np.zeros(self.config.test_trial)
        nav_error = np.zeros(self.config.test_trial)
        for i in range(self.config.test_trial):
            vrep.simxFinish(-1)
            env = Env(self.goal)
            if env.client != -1:
                print("Connected to V-REP server")
                state = env.state
                vrep.simxSynchronousTrigger(env.client)
                done = False
                while steps[i] < self.config.sim_time:
                    if not done:
                        reward, state, done = env.step(self.action(state))
                        rewards[i] += reward
                        steps[i] += 1
                        path[i] = env.path
                        nav_error[i] = env.current_error
                    else:
                        vrep.simxStopSimulation(
                            env.client, vrep.simx_opmode_oneshot)
                        if vrep.simxGetConnectionId(env.client) == -1:
                            break
            else:
                print("Couldn't connect to V-REP server!")
        return [rewards, steps, path, nav_error]

    def noise_action(self, state):
        noise = self.ou_noise.noise()
        return np.clip(self.actor.action(state) + noise, -1.0, 1.0)

    def action(self, state):
        return self.actor.action(state)

    def buffer_size(self):
        return self.memory.size

    def _build_summary(self):
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
