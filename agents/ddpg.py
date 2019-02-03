import numpy as np
import tensorflow as tf

from agents.networks import Actor
from agents.networks import Critic

from utility import ReplayMemory


class DDPG(object):
    def __init__(self, sess, state_dim, action_dim, u_bound, critic_lr, actor_lr, n_step, critic_l2,
                 clip_norm, tau, use_layer_norm, use_noisynet, noise, gamma, memory_size,
                 batch_size):
        self.gamma = gamma
        self.n_step = n_step

        self._sess = sess

        self._critic_l2 = critic_l2
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._clip_norm = clip_norm

        self._use_noisynet = use_noisynet
        self._tau = tau
        self._batch_size = batch_size
        self._action_bound = u_bound

        self._global_step = tf.train.get_or_create_global_step()

        if not use_noisynet:
            self._noise = noise
        self._memory = ReplayMemory(memory_size, batch_size, state_dim, action_dim)

        with tf.variable_scope('inputs'):
            self._is_training = tf.placeholder(tf.bool, name='is_training')
            self._observation_tf = tf.placeholder(tf.float32, [None, state_dim], name='state')
            self._action_tf = tf.placeholder(tf.float32, [None, action_dim], name='action')
            self._t_observation_tf = tf.placeholder(tf.float32, [None, state_dim],
                                                    name='target_state')

        with tf.variable_scope('actor'):
            self._actor = Actor(action_dim, self._action_bound[1], self._is_training,
                                use_layer_norm, use_noisynet)
            self._pi_tf = self._actor(self._observation_tf, 'main')
            self._target_pi_tf = self._actor(self._t_observation_tf, 'target')

        with tf.variable_scope('critic'):
            self._critic = Critic(self._is_training, use_layer_norm, use_noisynet)
            self._q_value_tf = self._critic(self._observation_tf, self._action_tf, 'main')
            self._q_value_pi_tf = self._critic(self._observation_tf, self._pi_tf, 'main', True)
            self._target_q_value_tf = self._critic(self._t_observation_tf, self._target_pi_tf,
                                                   'target')

        self._build_train_method()
        self._update_target_op = self._soft_update_target_networks()

    def act(self, state, explore=False):
        pi, q = self._sess.run([self._pi_tf, self._q_value_pi_tf], feed_dict={
            self._observation_tf: [state],
            self._is_training: False,
        })

        if not self._use_noisynet and explore:
            noise = self._noise() * np.mean(self._action_bound)
            pi[0] += noise
            pi[0] = np.clip(pi[0], self._action_bound[0], self._action_bound[1])

        return pi[0].copy(), q[0].copy()

    def hard_update_target_networks(self):
        self._sess.run([
            t_var.assign(var) for var, t_var in
            zip(self.main_trainable_vars, self.target_trainable_vars)
        ])

    def observe(self, state, action, reward, next_state, done):
        self._memory.push(state, action, reward, next_state, done)

        if done and not self._use_noisynet:
            self._noise.reset()

        if self._memory.size >= self._batch_size:
            self._train_mini_batch()

    def _build_train_method(self):
        with tf.variable_scope('optimizer'):
            with tf.variable_scope('critic'):
                self._build_critic_train_method()

            with tf.variable_scope('actor'):
                self._build_actor_train_method()

    def _build_critic_train_method(self):
        self._reward_tf = tf.placeholder(tf.float32, [None, 1], 'reward')
        self._done_tf = tf.placeholder(tf.float32, [None, 1], 'terminal')

        target_y = (self._reward_tf + (1.0 - self._done_tf)
                    * self.gamma ** self.n_step * self._target_q_value_tf)

        self._critic_loss = tf.reduce_mean(tf.square(target_y - self._q_value_tf))
        if self._critic_l2 > 0.0:
            w_l2 = [var for var in self._get_trainable_vars('critic', 'main')
                    if 'kernel' in var.name and 'output' not in var.name]
            reg = tf.contrib.layers.l2_regularizer(self._critic_l2)
            l2_loss = tf.contrib.layers.apply_regularization(reg, w_l2)
            self._critic_loss += l2_loss

        grads = tf.gradients(self._critic_loss, self._get_trainable_vars('critic', 'main'))
        if self._clip_norm > 0.0:
            grads, _ = tf.clip_by_global_norm(grads, self._clip_norm)
        optim = tf.train.AdamOptimizer(self._critic_lr)
        self._critic_train_op = optim.apply_gradients(
            zip(grads, self._get_trainable_vars('critic', 'main')), global_step=self._global_step)

    def _build_actor_train_method(self):
        self._actor_loss = -tf.reduce_mean(self._q_value_pi_tf)
        grads = tf.gradients(self._actor_loss, self._get_trainable_vars('actor', 'main'))
        if self._clip_norm > 0.0:
            grads, _ = tf.clip_by_global_norm(grads, self._clip_norm)
        optim = tf.train.AdamOptimizer(self._actor_lr)
        self._actor_train_op = optim.apply_gradients(
            zip(grads, self._get_trainable_vars('actor', 'main')))

    def _train_mini_batch(self):
        train_batch = self._memory.sample()
        self._sess.run(
            [self._critic_train_op, self._actor_train_op], feed_dict={
                self._reward_tf: np.vstack(train_batch['r']).astype(np.float32),
                self._done_tf: np.vstack(train_batch['d']).astype(np.float32),
                self._observation_tf: np.vstack(train_batch['obs1']).astype(np.float32),
                self._action_tf: np.vstack(train_batch['u']).astype(np.float32),
                self._t_observation_tf: np.vstack(train_batch['obs2']).astype(np.float32),
                self._is_training: True,
            })

        self._sess.run(self._update_target_op)

    def _soft_update_target_networks(self):
        with tf.variable_scope('update_targets'):
            update = [
                t_var.assign(t_var * (1.0 - self._tau) + var * self._tau)
                for var, t_var in zip(self.main_trainable_vars, self.target_trainable_vars)
            ]
        return update

    @property
    def global_step(self):
        return self._global_step.eval()

    @property
    def main_trainable_vars(self):
        name = 'main'
        return self._get_trainable_vars('critic', name) + self._get_trainable_vars('actor', name)

    @property
    def target_trainable_vars(self):
        name = 'target'
        return self._get_trainable_vars('critic', name) + self._get_trainable_vars('actor', name)

    @property
    def is_warm_up(self):
        return self._memory.size < self._batch_size

    @staticmethod
    def _get_global_vars(type, name):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=type + '/' + name)

    @staticmethod
    def _get_trainable_vars(type, name):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=type + '/' + name)
