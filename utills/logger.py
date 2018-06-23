import os

import numpy as np
import tensorflow as tf

import utills.opts as U

class Logger(object):
    """
    Logger class for reinforcement learning agent based on TensorFLow.
    """
    def __init__(self, sess, agent, logdir):
        """
        :param sess: TensorFlow session
        :param agent: object, rl agent
        :param logdir: string, path for logs
        """
        self._sess = sess
        self._logdir = logdir
        self._agent = agent

        with tf.variable_scope('summary'):
            self._build_variables_sumamry()
            self._build_test_summary()

        self._merged = tf.summary.merge_all()

        self._train_writer = tf.summary.FileWriter(
            os.path.join(logdir, 'train'), sess.graph)
        self._test_writer = tf.summary.FileWriter(
            os.path.join(logdir, 'test'))

        self.saver = None

    def save(self, global_step):
        """
        Save model
        :param global_step: integer or tf variable, step of training
        """
        self.saver.save(self._sess, os.path.join(self._logdir, 'model'),
                        global_step=global_step)

    def writer(self, ep_reward, ep_q, ep_step, test=False):
        """
        Write Tensorboard logs from episodes
        :param ep_reward: list, reward values
        :param ep_q: list, q values
        :param ep_step: list, episode steps
        :param test, boolean, determine if testing episode
        """
        summary = self._sess.run(self._merged, feed_dict={
            self._ep_reward: self._check_type(self._ep_reward, ep_reward),
            self._ep_q: self._check_type(self._ep_q, ep_q),
            self._ep_steps: self._check_type(self._ep_steps, ep_step),
        })
        if test:
            self._test_writer.add_summary(summary, self._agent.global_step)
        else:
            self._train_writer.add_summary(summary, self._agent.global_step)

    def load(self):
        """
        Load model variables from last checkpoint
        """
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self._logdir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self._sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def _build_variables_sumamry(self):
        for var in self._agent.main_trainable_vars:
            tf.summary.histogram(var.name, var)

        for var in self._agent.target_trainable_vars:
            tf.summary.histogram(var.name, var)

    def _build_test_summary(self):
        self._ep_reward = tf.placeholder(
            tf.float32, [None, ], name='episode_rewards')
        self._ep_q = tf.placeholder(
            tf.float32, [None, ], name='episode_q')
        self._ep_steps = tf.placeholder(
            tf.float32, [None, ], 'episode_steps')
        tf_map = {
            'reward_summary': self._ep_reward,
            'q_value_summary': self._ep_q,
            'step_summary': self._ep_steps
        }

        for key, var in tf_map.items():
            with tf.variable_scope(key):
                tf.summary.scalar('mean', tf.reduce_mean(var, axis=0))
                tf.summary.scalar('std', U.reduce_std(var, axis=0))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.scalar('max', tf.reduce_max(var))

    @staticmethod
    def _check_type(ref_var, var):
        if not ref_var.get_shape().is_compatible_with(np.shape(var)):
            var = [var]

        return var


def env_logger(env):
    """
    Print gym env parameters
    :param env: gym.env,
    """
    u_low = env.action_space.low
    u_high = env.action_space.high
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    max_steps = env._max_episode_steps

    print('Environment information')
    print('-----------------------')
    print('Action: \n low: {} \n high: {}'.format(u_low, u_high))
    print('-----------------------')
    print('Observation: \n low: {} \n high: {}'.format(obs_low, obs_high))
    print('-----------------------')
    print('Episode max steps: {}'.format(max_steps))
    print('-----------------------')
