import os
import tensorflow as tf

from ddpg import DDPGAgent
from utils.config import AgentConfig


def main(state):
    goal = (2.0, 2.0)
    config = AgentConfig
    ddpg = DDPGAgent(goal, config)
    if state == 'train':
        ddpg.train()
    else:
        ddpg.play()

if __name__ == '__main__':
    tf.reset_default_graph()
    os.system('clear')
    main()
