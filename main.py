import argparse
import inspect

import gym
import gym_vrep
import tensorflow as tf

from ddpg import DDPG
from utills.play import Play


def parser_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--logdir', type=str, default='output')
    parser.add_argument('--env_id', type=str,
                        default='MobileRobotIdealNavigation-v0')
    parser.add_argument('--env_dt', type=float, default=0.05)
    parser.add_argument('--nb_episodes', type=int, default=1000)
    parser.add_argument('--nb_eval_episodes', type=int, default=10)
    parser.add_argument('--exploration', type=int, default=100000)
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--critic_l2', type=float, default=1e-3)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=10.0)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--noisy_layer', action='store_true')

    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


def main(env_id, train, logdir, **kwargs):
    sess = tf.InteractiveSession()
    env = gym.make(env_id)

    u_bound = {
        'low': env.action_space.low,
        'high': env.action_space.high,
    }
    sig = inspect.signature(DDPG)
    ddpg_kwargs = dict()
    for key in sig.parameters:
        if key in kwargs:
            ddpg_kwargs[key] = kwargs[key]
            kwargs.pop(key)

    agent = DDPG(sess, env.observation_space.shape[0], env.action_space.shape[0], u_bound=u_bound,
                 **ddpg_kwargs)

    play = Play(sess, env, agent, logdir)

    if train:
        play.train(kwargs['nb_episodes'], kwargs['nb_eval_episodes'])

    play.eval()
    env.close()


if __name__ == '__main__':
    tf.reset_default_graph()
    args = parser_setup()
    main(**args)
