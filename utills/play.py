import numpy as np
import tensorflow as tf
import utills.opts as U

from tqdm import tqdm
from utills.logger import Logger


class Play(object):
    def __init__(self, sess, env, agent, logdir):
        self._sess = sess
        self._env = env
        self._agent = agent
        self._logger = Logger(self._sess, self._agent, logdir)
        self._eval_seeds = [42, 125, 356, 466, 689, 1024, 1337, 2490, 3809, 6667]

        self._sess.run(tf.global_variables_initializer())
        self._logger.load()
        self._agent.initialize_target_networks()

    def train(self, nb_episodes, nb_eval_episodes):
        reward_for_save = 0.0
        for ep in tqdm(range(nb_episodes)):
            self._set_seed(5)
            self._agent.ou_noise.reset()
            ep_score, ep_q_values, ep_step = self._run_env(True)

            self._logger.writer(ep_score, ep_q_values, ep_step)
            if ep % nb_eval_episodes == (nb_eval_episodes - 1):
                eval_scores = list()
                eval_q_values = list()
                eval_steps = list()
                for seed in self._eval_seeds:
                    self._set_seed(seed)
                    score, q_values, steps = self._run_env(False)
                    eval_scores.append(score)
                    eval_q_values.append(q_values)
                    eval_steps.append(steps)
                self._logger.writer(eval_scores, np.hstack(eval_q_values), eval_steps, True)

                if np.mean(eval_scores) > reward_for_save:
                    info = "Saved model, global step {}, test reward {}".format(
                        self._agent.global_step, np.mean(eval_scores))
                    print(info)
                    self._logger.save(self._agent.global_step)
                    reward_for_save = np.mean(eval_scores)
            pass

    def eval(self):
        self._run_env(False)

    def _run_env(self, train):
        episode_q_values = list()
        score = 0.0
        step = 0.0
        state = self._env.reset()
        for i in range(self._env._max_episode_steps):
            step = i
            action, q_value = self._agent.act(state, train)
            action = self._set_action(action)
            next_state, reward, terminal, _ = self._env.step(action)
            if train:
                self._agent.observe(state, action, reward, next_state, terminal)

            state = next_state
            score += reward
            episode_q_values.append(q_value)
            if terminal:
                break
        return score, np.squeeze(episode_q_values), step

    def _set_seed(self, seed):
        self._env.seed(seed)
        tf.random.set_random_seed(seed)
        np.random.seed(seed)

    def _set_action(self, action):
        return U.scale(action, -1.0, 1.0, self._env.action_space.low, self._env.action_space.high)
