import tensorflow as tf

from ddpg import DDPGAgent
from environment.env import Env
from utils.config import AgentConfig
from vrep import vrep

def main():
    goal = (2.0, 2.0)
    config = AgentConfig
    env = Env(config, goal)
    tf.reset_default_graph()

    with tf.Session() as sess:
        for _ in range(config.num_episode):
            client = env.run_env()
            ddpg = DDPGAgent(sess, goal, config)
            if client != -1:
                state = env.reset(client)
                while vrep.simxGetConnectionId(client) != -1:
                    action = ddpg.noise_action(state)
                    reward, next_state, done = env.step(action)
                    ddpg.observe(state, action, reward, next_state, done)
                    if done:
                        vrep.simxStopSimulation(client, vrep.simx_opmode_oneshot)
                        break

if __name__ == '__main__':
    main()
