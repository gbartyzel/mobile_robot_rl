import numpy as np
import tensorflow as tf

from networks import Actor
from networks import Critic

import utills.opts as U
from utills.ounoise import OUNoise
from utills.memory import ReplayMemory


class DDPG(object):
    def __init__(self, sess, dimo, dimu, o_bound, u_bound, critic_lr, actor_lr,
                 critic_l2, clip_norm, tau, layer_norm, noisy_layer, gamma,
                 memory_size, exploration, batch_size, env_dt, norm):
        self._sess = sess

        self._dimo = dimo
        self._dimu = dimu
        self._critic_l2 = critic_l2
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._clip_norm = clip_norm

        self._noisy = noisy_layer
        self._gamma = gamma
        self._tau = tau
        self._batch_size = batch_size
        self._u_bound = u_bound
        self._o_bound = o_bound
        self._norm = norm

        self._global_step = tf.train.get_or_create_global_step()

        self.ou_noise = OUNoise(
            dim=dimu, n_step_annealing=exploration, dt=env_dt)
        self._memory = ReplayMemory(memory_size)

        with tf.variable_scope('inputs'):
            self._is_training = tf.placeholder(tf.bool, name='is_training')
            self._obs = tf.placeholder(
                tf.float32, [None, self._dimo], name='state')
            self._u = tf.placeholder(
                tf.float32, [None, self._dimu], name='action')
            self._t_obs = tf.placeholder(
                tf.float32, [None, self._dimo], name='target_state')

        with tf.variable_scope('actor'):
            self._actor = Actor(
                'main', self._obs, dimu, self._is_training, layer_norm,
                noisy_layer)
            self._target_actor = Actor(
                'target', self._t_obs, dimu, self._is_training, layer_norm,
                noisy_layer)

        with tf.variable_scope('critic'):
            self._critic = Critic(
                'main', self._obs, self._u, self._is_training, layer_norm,
                noisy_layer)
            self._critic_pi = Critic(
                'main', self._obs, U.scaling(
                    self._actor.pi, 0.0, 1.0, u_bound['low'], u_bound['high']),
                self._is_training, layer_norm, noisy_layer, reuse=True)
            self._target_critic = Critic(
                'target', self._t_obs,
                U.scaling(self._target_actor.pi, 0.0, 1.0, u_bound['low'],
                          u_bound['high']), self._is_training, layer_norm,
                noisy_layer)

        self._build_train_method()
        self._update_target_op = self._update_target_networks()

    @property
    def global_step(self):
        return self._global_step.eval()

    @property
    def main_trainable_vars(self):
        return self._critic.trainable_vars + self._actor.trainable_vars

    @property
    def target_trainable_vars(self):
        return (self._target_critic.trainable_vars +
                self._target_actor.trainable_vars)

    def act_noisy(self, state):
        pi, q = self._sess.run(
            [self._actor.pi, self._critic_pi.Q], feed_dict={
                self._obs: [state],
                self._is_training: True,
            })
        if not self._noisy:
            pi[0] += self.ou_noise.noise()
        pi = np.clip(pi[0], 0.0, 1.0)
        pi = U.scaling(
            pi, 0.0, 1.0, self._u_bound['low'], self._u_bound['high'])
        return pi.copy(), q[0].copy()

    def act(self, state):
        pi, q = self._sess.run(
            [self._actor.pi, self._critic_pi.Q], feed_dict={
                self._obs: [state],
                self._is_training: False,
            })
        pi = U.scaling(
            pi[0], 0.0, 1.0, self._u_bound['low'], self._u_bound['high'])
        return pi.copy(), q[0].copy()

    def initialize_target_networks(self):
        self._sess.run([
            t_var.assign(var) for var, t_var in
            zip(self.main_trainable_vars, self.target_trainable_vars)
        ])

    def observe(self, state, action, reward, next_state, done):
        self._memory.add(state, action, reward, next_state, done)

        if self._memory.size >= self._batch_size:
            self._train_mini_batch()

    def _build_train_method(self):
        with tf.variable_scope('optimizer'):
            self._reward = tf.placeholder(tf.float32, [None, 1], 'reward')
            self._done = tf.placeholder(tf.float32, [None, 1], 'terminal')
            target_y = (self._reward + (1.0 - self._done) * self._gamma
                        * self._target_critic.Q)

            self._critic_loss = tf.reduce_mean(
                tf.square(tf.stop_gradient(target_y) - self._critic.Q))
            if self._critic_l2 > 0.0:
                w_l2 = [var for var in self._critic.trainable_vars
                        if 'kernel' in var.name and 'output' not in var.name]
                reg = tf.contrib.layers.l2_regularizer(self._critic_l2)
                l2_loss = tf.contrib.layers.apply_regularization(reg, w_l2)
                self._critic_loss += l2_loss

            c_grads = tf.gradients(
                self._critic_loss, self._critic.trainable_vars)
            if self._clip_norm > 0.0:
                c_grads, _ = tf.clip_by_global_norm(c_grads, self._clip_norm)
            critic_optim = tf.train.AdamOptimizer(self._critic_lr)
            self._critic_train_op = critic_optim.apply_gradients(
                zip(c_grads, self._critic.trainable_vars),
                global_step=self._global_step)

            self._actor_loss = -tf.reduce_mean(self._critic_pi.Q)
            a_grads = tf.gradients(self._actor_loss,
                                   self._actor.trainable_vars)
            if self._clip_norm > 0.0:
                a_grads, _ = tf.clip_by_global_norm(a_grads, self._clip_norm)
            actor_optim = tf.train.AdamOptimizer(self._actor_lr)
            self._actor_train_op = actor_optim.apply_gradients(
                zip(a_grads, self._actor.trainable_vars))

    def _train_mini_batch(self):
        train_batch = self._memory.sample(self._batch_size)
        state_batch = np.vstack(train_batch['obs1']).astype(np.float32)
        action_batch = np.vstack(train_batch['u']).astype(np.float32)
        reward_batch = np.vstack(train_batch['r']).astype(np.float32)
        next_state_batch = np.vstack(train_batch['obs2']).astype(np.float32)
        done_batch = np.vstack(train_batch['d']).astype(np.float32)

        self._sess.run(
            [self._critic_train_op, self._actor_train_op], feed_dict={
                self._reward: reward_batch,
                self._done: done_batch,
                self._obs: state_batch,
                self._u: action_batch,
                self._t_obs: next_state_batch,
                self._is_training: True,
            })

        self._sess.run(self._update_target_op)

    def _update_target_networks(self):
        with tf.variable_scope('update_targets'):
            update = [
                t_var.assign(
                    t_var * (1.0 - self._tau) + var * self._tau)
                for var, t_var in
                zip(self.main_trainable_vars, self.target_trainable_vars)
            ]
        return update
