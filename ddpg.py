import numpy as np
import tensorflow as tf

from tqdm import tqdm

from environment.env import Env
from networks.actor import Actor
from networks.critic import Critic
from utils.memory import ReplayMemory
from utils.ounoise import OUNoise


class DDPGAgent(object):
    def __init__(self, FLAGS):
        self._env = Env(
            FLAGS.env, visulalization=FLAGS.viz, normalization=FLAGS.norm)
        self._flags = FLAGS

        self._sess = tf.InteractiveSession()

        self._critic = Critic(self._sess, self._env)
        self._actor = Actor(self._sess, self._env)

        self._avg_reward,\
        self._min_reward,\
        self._max_reward,\
        self._avg_steps = self._build_summary()

        self._merged = tf.summary.merge_all()
        self._writer = tf.summary.FileWriter(self._flags.summary_path,
                                             self._sess.graph)

        self._sess.run(tf.global_variables_initializer())

        self._actor.update_target_network('copy')
        self._critic.update_target_network('copy')

        self._saver = self._load()
        self._ou_noise = OUNoise(
            action_dim=self._env.action_dim,
            n_step_annealing=self._flags.exploration,
            dt=self._env.dt)

        self._memory = ReplayMemory(self._flags.memory_size)

    def _run(self, mode):
        ret_reward = 0.0
        ret_step = 0
        state = self._env.reset()
        for step in range(self._env.max_steps):
            action = self._action(state, mode)
            reward, next_state, done = self._env.step(action)
            if mode == 'train':
                self._observe(state, action, reward, next_state, done)
            state = next_state
            ret_reward += reward
            ret_step = step
            if done:
                self._env.stop()
                break

        return ret_reward, ret_step

    def train(self):
        for ep in tqdm(list(range(self._flags.episodes))):
            self._run('train')

            if (ep % self._flags.test_episodes == self._flags.test_episodes - 1
                    and self._memory.size >= self._flags.warm_up):
                ret_rewards, ret_steps = self.test()

                summary = self._sess.run(
                    self._merged,
                    feed_dict={
                        self._avg_reward: np.mean(ret_rewards),
                        self._min_reward: np.min(ret_rewards),
                        self._max_reward: np.max(ret_rewards),
                        self._avg_steps: np.mean(ret_steps),
                    })
                self._writer.add_summary(summary,
                                         self._critic.global_step.eval())

    def test(self):
        test_rewards = np.zeros(self._flags.trials)
        test_steps = np.zeros(self._flags.trials)
        for i in range(self._flags.trials):
            test_rewards[i], test_steps[i] = self._run('test')
        return [test_rewards, test_steps]

    def _action(self, state, mode):
        if mode == 'train':
            noise = self._ou_noise.noise()
            action = np.clip(
                self._actor.prediction([state])[0] + noise, -1.0, 1.0)
        elif mode == 'test':
            action = self._actor.prediction([state])[0]

        return action

    def _build_summary(self):
        with tf.variable_scope('summary'):
            avg_reward = tf.placeholder(tf.float32)
            min_reward = tf.placeholder(tf.float32)
            max_reward = tf.placeholder(tf.float32)
            avg_steps = tf.placeholder(tf.float32)
            tf.summary.scalar('avg_reward', avg_reward)
            tf.summary.scalar('max_reward', max_reward)
            tf.summary.scalar('min_reward', min_reward)
            tf.summary.scalar('avg_steps', avg_steps)

        return avg_reward, min_reward, max_reward, avg_steps

    def _load(self):
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self._flags.model_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self._sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded:', checkpoint.model_checkpoint_path)
        else:
            print('Could not find old network weights')
        return saver

    def _observe(self, state, action, reward, next_state, done):
        self._memory.add(state, action, reward, next_state, done)

        if self._memory.size >= self._flags.warm_up:
            self._train_mini_batch()

        if self._critic.global_step.eval() % self._flags.warm_up == 0:
            self._saver.save(
                self._sess,
                self._flags.model_path + '/model',
                global_step=self._critic.global_step)

        if done:
            self._ou_noise.reset()

    def _train_mini_batch(self):
        train_batch = self._memory.sample(self._flags.batch_size)

        state_batch = np.asarray([data[0] for data in train_batch])
        action_batch = np.asarray([data[1] for data in train_batch])
        reward_batch = np.asarray([data[2] for data in train_batch])
        next_state_batch = np.asarray([data[3] for data in train_batch])
        done_batch = [data[4] for data in train_batch]

        reward_batch = np.resize(reward_batch, [self._flags.batch_size, 1])
        done_batch = np.resize(done_batch, [self._flags.batch_size, 1])

        action_batch = np.resize(
            action_batch, [self._flags.batch_size, self._env.action_dim])

        next_action = self._actor.target_prediction(next_state_batch)

        q_value = self._critic.target_prediction(next_state_batch, next_action)
        y_batch = np.zeros((self._flags.batch_size, 1))

        for i in range(self._flags.batch_size):
            if done_batch[i]:
                y_batch[i] = reward_batch[i]
            else:
                y_batch[i] = (
                    self._flags.discount * q_value[i] + reward_batch[i])
        self._critic.train(state_batch, action_batch, y_batch)

        gradient_actions = self._actor.prediction(state_batch)
        q_gradients = self._critic.gradients(state_batch, gradient_actions)
        self._actor.train(state_batch, q_gradients)

        self._actor.update_target_network()
        self._critic.update_target_network()
