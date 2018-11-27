import argparse
import inspect

import gym
import gym_vrep
import numpy as np
import tensorflow as tf

from utills.logger import env_logger, Logger
from utills.opts import scaling

from ddpg import DDPG
from tqdm import tqdm


def parser_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--logdir', type=str, default='output')
    parser.add_argument('--env_id', type=str,
                        default='MobileRobotIdealNavigation-v0')
    parser.add_argument('--env_dt', type=float, default=0.05)
    parser.add_argument('--nb_episodes', type=int, default=10000)
    parser.add_argument('--nb_eval_episodes', type=int, default=10)
    parser.add_argument('--nb_trials', type=int, default=5)
    parser.add_argument('--exploration', type=int, default=1000000)
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--critic_l2', type=float, default=1e-2)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--noisy_layer', action='store_true')
    parser.add_argument('--norm', action='store_true')

    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


def play_env(env, agent, train=False):
    qs = []
    rewards = 0.0
    step = 0.0
    if not train:
        env.seed()
    s_t = env.reset()
    for i in range(env._max_episode_steps):
        step = i
        if train:
            a_t, q_t = agent.noisy_action(s_t)
        else:
            a_t, q_t = agent.action(s_t)
        s_t_1, r_t, d_t, _ = env.step(a_t)
        if train:
            agent.observe(s_t, a_t, r_t, s_t_1, d_t)
        s_t = s_t_1
        rewards += r_t
        qs.append(q_t)
        if d_t:
            break
    agent.ou_noise.reset()

    return np.squeeze(rewards), np.squeeze(qs), step


def training(env, agent, logger, nb_episodes, nb_eval_episodes, nb_trials):
    reward_for_save = 0.0
    for ep in tqdm(range(nb_episodes)):
        ep_r, ep_q, ep_step = play_env(env, agent, True)
        logger.writer(ep_r, ep_q, ep_step)
        if ep % nb_eval_episodes == 0:
            train_r = []
            train_q = []
            train_step = []
            for _ in range(nb_trials):
                r, q, step = play_env(env, agent)
                train_r.append(r)
                train_q.append(q)
                train_step.append(step)
            logger.writer(train_r, np.hstack(train_q), train_step, True)

            if np.mean(train_r) > reward_for_save:
                info = "Saved model, global step {}, test reward {}".format(
                    agent.global_step, np.mean(train_r))
                print(info)
                logger.save(agent.global_step)
                reward_for_save = np.mean(train_r)


def main(env_id, train, play, logdir, **kwargs):
    sess = tf.InteractiveSession()
    env = gym.make(env_id)

    env_logger(env)

    dimu = env.action_space.shape[0]
    dimo = env.observation_space.shape[0]
    u_bound = {
        'low': env.action_space.low,
        'high': env.action_space.high,
    }
    o_bound = {
        'low': env.observation_space.low,
        'high': env.observation_space.high,
    }
    sig = inspect.signature(DDPG)
    ddpg_kwargs = dict()
    for key in sig.parameters:
        if key in kwargs:
            ddpg_kwargs[key] = kwargs[key]
            kwargs.pop(key)

    agent = DDPG(sess, dimo, dimu, o_bound=o_bound, u_bound=u_bound,
                 **ddpg_kwargs)
    logger = Logger(sess, agent, logdir)

    sess.run(tf.global_variables_initializer())

    logger.load()

    agent.initialize_target_networks()

    if train:
        training(env, agent, logger, **kwargs)

    if play:
        play_env(env, agent)

    env.close()


if __name__ == '__main__':
    tf.reset_default_graph()
    args = parser_setup()
    main(**args)
