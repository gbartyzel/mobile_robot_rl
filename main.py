import argparse
import sys

import gym
import gym_vrep
import tensorflow as tf

from ddpg import DDPGAgent
from networks.actor import Actor
from networks.critic import Critic

def parser_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model_path', type=str, default='saved_model')
    parser.add_argument('--summary_path', type=str, default='train_output')
    parser.add_argument('--env_id', type=str,
                        default='VrepMobileRobotIdealNavigation-v0')
    parser.add_argument('--nb_episodes', type=int, default=1000)
    parser.add_argument('--nb_eval_episodes', type=int, default=10)
    parser.add_argument('--nb_trials', type=int, default=5)
    parser.add_argument('--warm_up', type=int, default=10000)
    parser.add_argument('--exploration', type=int, default=1000000)
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    parser.add_argument('--critic_l2', type=float, default=1e-2)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float)

    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


def main(env_id, train, test, **kwargs):
    env = gym.make(env_id)




if __name__ == '__main__':
    tf.reset_default_graph()
    args = parser_setup()
    main(**args)

