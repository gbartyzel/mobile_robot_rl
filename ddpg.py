import os

import numpy as np
import tensorflow as tf

from networks import Actor
from networks import Critic
from utils.memory import ReplayMemory
from utils.ounoise import OUNoise


class DDPGAgent(object):
    def __init__(self, sess, dimo, dimu, critic_lr, actor_lr, l2term,
                 clip_norm, tau, batch_norm, noisy_layer, gamma, memory_size,
                 exploration, batch_size, dt, logdir):
        self._sess = sess

        self._dimo = dimo
        self._dimu = dimu
        self._l2term = l2term
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._clip_norm = clip_norm

        self._gammma = gamma

        self._global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope('inputs'):
            self._obs = tf.placeholder(
                tf.float32, [None, self._dimo], name='input_state')
            self._u = tf.placeholder(
                tf.float32, [None, self._dimu], name='input_action')
            self._is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.variable_scope('actor'):
            self._actor = Actor(
                'main', self._obs, self._is_training, dimu, batch_norm,
                noisy_layer)
            self._target_actor = Actor(
                'target', self._obs, self._is_training, dimu, batch_norm,
                noisy_layer)

        with tf.variable_scope('critic') as vs:
            self._critic = Critic(
                'main', self._obs, self._u, self._is_training, batch_norm,
                noisy_layer)
            self._critic_pi = Critic(
                'main', self._obs, self._actor.pi, self._is_training,
                batch_norm, noisy_layer, reuse=True)
            self._target_critic = Critic(
                'target', self._obs, self._u, self._is_training, batch_norm,
                noisy_layer)
        self._build_train_method()

        self._merged = tf.summary.merge_all()
        self._writer = tf.summary.FileWriter(
            os.path.join(logdir, 'log'), self._sess.graph)

        self._sess.run(tf.global_variables_initializer())

        self._saver = self._load(logdir)
        self._ou_noise = OUNoise(
            action_dim=dimu, n_step_annealing=exploration, dt=dt)
        print(self._critic.trainable_vars)

        self._memory = ReplayMemory(memory_size)

    def _load(self, path):
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self._sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        return saver

    def _build_train_method(self):
        with tf.variable_scope('optimizer'):
            self._reward = tf.placeholder(tf.float32, [None], 'input_reward')
            self._done = tf.placeholder(tf.float32, [None], 'input_done')
            target_y = tf.add(
                self._reward, tf.multiply(tf.multiply(
                    self._target_critic.Q, self._gammma), 1.0 - self._done))

            self._critic_loss = tf.losses.mean_squared_error(
                tf.stop_gradient(target_y), self._critic.Q,
                reduction=tf.losses.Reduction.MEAN)
            if self._l2term:
                l2_loss = tf.add_n(
                    [
                        self._l2term * tf.nn.l2_loss(var)
                        for var in self._critic.trainable_vars
                        if 'kernel' in var.name and 'output' not in var.name
                    ], name='l2_loss')
                self._critic_loss += l2_loss

            grads = tf.gradients(
                self._critic_loss, self._critic.trainable_vars)
            if self._clip_norm:
                grads, _ = tf.clip_by_global_norm(grads, self._clip_norm)
            critic_optim = tf.train.AdamOptimizer(self._critic_lr)
            self._critic_train_op = critic_optim.apply_gradients(
                zip(grads, self._critic.trainable_vars),
                global_step=self._global_step)

            self._actor_loss = -tf.reduce_mean(self._critic_pi.Q)
            grads = tf.gradients(self._actor_loss,
                                 self._actor.trainable_vars)
            if self._clip_norm:
                grads, _ = tf.clip_by_global_norm(grads, self._clip_norm)
            actor_optim = tf.train.AdamOptimizer(self._actor_lr)
            self._actor_train_op = actor_optim.apply_gradients(
                zip(grads, self._actor.trainable_vars))

    """
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
    """


param = {
    'dimo': 7,
    'dimu': 2,
    'critic_lr': 1e-3,
    'actor_lr': 1e-4,
    'l2term': 1e-2,
    'clip_norm': None,
    'tau': 1e-3,
    'batch_norm': False,
    'noisy_layer': False,
    'gamma': 0.99,
    'memory_size': int(1e6),
    'exploration': int(1e6),
    'batch_size': 64,
    'dt': 0.05,
    'logdir': 'output',

}
with tf.Session() as sess:
    model = DDPGAgent(sess, **param)
