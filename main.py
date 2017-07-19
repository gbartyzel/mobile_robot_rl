import os
import tensorflow as tf

from ddpg import DDPGAgent
from utils.config import AgentConfig


def main():
    goal = (2.0, 2.0)
    config = AgentConfig
    ddpg = DDPGAgent(goal, config)
    ddpg.train()

if __name__ == '__main__':
    tf.reset_default_graph()
    os.system('clear')
    main()
