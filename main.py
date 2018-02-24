import argparse
import sys
import tensorflow as tf

#from ddpg import DDPGAgent
from environment.env import Env
FLAGS = None


def main(_):
    env = Env("room", visulalization=True)
    env.reset()
    """
    ddpg = DDPGAgent(FLAGS)
    if FLAGS.train:
        ddpg.train()
    else:
        ddpg.play()
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true', help='Enable training mode')
    parser.add_argument(
        '--test', action='store_true', help='Enable testing mode')
    parser.add_argument(
        '--model_path',
        type=str,
        default='saved_ddpg',
        help='Path for model files')
    parser.add_argument(
        '--summary_path',
        type=str,
        default='train_output',
        help='Path for Tensorboard summary')
    parser.add_argument(
        '--env',
        type=str,
        default='room',
        help='Name of the environment to load')
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Max number of episodes during learning')
    parser.add_argument(
        '--test_episodes',
        type=int,
        default=10,
        help='Frequency of model testing')
    parser.add_argument(
        '--trails',
        type=int,
        default=5,
        help='Number of trails during testing')
    parser.add_argument(
        '--warm_up',
        type=int,
        default=10000,
        help='Number of steps before learning starts')
    parser.add_argument(
        '--exploration',
        type=int,
        default=1000000,
        help='Define maximum steps for exploration of the environment')
    parser.add_argument(
        '--memory_size',
        type=int,
        default=1000000,
        help='Define maximum capacity of replay buffer')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Define size of the minibatch')
    parser.add_argument(
        '--discount', type=float, default=0.99, help='Define discount factor')

    tf.reset_default_graph()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
