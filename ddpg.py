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
    def __init__(self, env, FLAGS):
        self._env = env
        self._FLAGS = FLAGS
        self._total_steps = 0

        self.sess = tf.InteractiveSession()

        self.critic = Critic(self.sess, FLAGS.batch_size)
        self.actor = Actor(self.sess)
        self._build_summary()

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.FLAGS.summary_path,
                                            self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.actor.update_target_network('copy')
        self.critic.update_target_network('copy')

        self.saver = self._load()
        self.ou_noise = OUNoise(action_dim=2)

        self.memory = ReplayMemory(self.FLAGS.memory_size)

        self.env = Env("room")

    def train(self):
        for ep in tqdm(list(range(self.FLAGS.episodes))):
            state = self.env.reset()
            done = False
            for step in range(self.env.max_steps):
                if not done:
                    action = self._action(state)
                    reward, next_state, done = self.env.step(action)
                    self._observe(state, action, reward, next_state, done)
                    state = next_state
                    self._total_steps += 1
                else:
                    if self.env.stop() == -1:
                        break

            if ep % self.FLAGS.test_episodes == self.FLAGS.test_episodes - 1:
                result = self.test()
                print(result)

                summary = self.sess.run(
                    self.merged,
                    feed_dict={
                        self.avg_reward: np.mean(result[0]),
                        self.min_reward: np.min(result[0]),
                        self.max_reward: np.max(result[0]),
                        self.avg_steps: np.mean(result[1]),
                    })
                self.writer.add_summary(summary, self._total_steps)

    def test(self):
        rewards = np.zeros(self.FLAGS.test_trials)
        steps = np.zeros(self.FLAGS.test_trials)
        for i in range(self.FLAGS.test_trials):
            state = self.env.reset()
            done = False
            step = 0
            ep_reward = 0.0
            """
                if not done:
                    step += 1
                    reward, state, done = env.norm_step(self._action(state))
                    ep_reward += reward
                    if step >= self.FLAGS.num_steps:
                        done = True
                else:
                    vrep.simxStopSimulation(env.client,
                                            vrep.simx_opmode_oneshot)
                    if vrep.simxGetConnectionId(env.client) == -1:
                        break
                rewards[i] = ep_reward
                steps[i] = step
            """
        return [rewards, steps]

    def _action(self, state):
        return self.actor.prediction([state])[0]

    def _noise_action(self, state):
        noise = self.ou_noise.noise()
        return np.clip(self.actor.prediction([state])[0] + noise, -1.0, 1.0)

    def _build_summary(self):
        with tf.variable_scope('summary'):
            self.avg_reward = tf.placeholder(tf.float32)
            self.min_reward = tf.placeholder(tf.float32)
            self.max_reward = tf.placeholder(tf.float32)
            self.avg_steps = tf.placeholder(tf.float32)
            tf.summary.scalar('avg_reward', self.avg_reward)
            tf.summary.scalar('max_reward', self.max_reward)
            tf.summary.scalar('min_reward', self.min_reward)
            tf.summary.scalar('avg_steps', self.avg_steps)

    def _load(self):
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.FLAGS.model_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        return saver

    def _observe(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if self.memory.size >= self.FLAGS.warm_up:
            self._train_mini_batch()

        if self._total_steps % self.FLAGS.warm_up == 0:
            self.saver.save(
                self.sess,
                self.FLAGS.model_path + '/model',
                global_step=self._total_steps)

        if done:
            self.ou_noise.reset()

    def _train_mini_batch(self):
        train_batch = self.memory.sample(self.FLAGS.batch_size)

        state_batch = np.asarray([data[0] for data in train_batch])
        action_batch = np.asarray([data[1] for data in train_batch])
        reward_batch = np.asarray([data[2] for data in train_batch])
        next_state_batch = np.asarray([data[3] for data in train_batch])
        done_batch = np.asarray(
            [data[4] for data in train_batch]).astype(float)

        reward_batch = np.resize(reward_batch, [self.FLAGS.batch_size, 1])
        done_batch = np.resize(done_batch, [self.FLAGS.batch_size, 1])

        action_batch = np.resize(action_batch, [self.FLAGS.batch_size, 2])

        next_action = self.actor.target_prediction(next_state_batch)

        q_value = self.critic.target_prediction(next_state_batch, next_action)

        y_batch = (1. - done_batch) * self.FLAGS.gamma * q_value + reward_batch

        _, loss = self.critic.train(state_batch, action_batch, y_batch)

        gradient_actions = self.actor.prediction(state_batch)
        q_gradients = self.critic.gradients(state_batch, gradient_actions)
        self.actor.train(state_batch, q_gradients)

        self.actor.update_target_network()
        self.critic.update_target_network()
