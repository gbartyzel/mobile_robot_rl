import os
import time
import math
import tensorflow as tf

from ddpg import DDPGAgent
from environment.env import Env
from utils.config import AgentConfig
from vrep import vrep


def normalize(state, max_error):
    state[0:-2] = state[0:-2] - 1.0
    state[-2] = (state[-2] - max_error / 2) / max_error / 2
    state[-1] = state[-1] / math.pi
    return state


def main():
    goal = (2.0, 2.0)
    config = AgentConfig
    tf.reset_default_graph()
    with tf.Session() as sess:
        ddpg = DDPGAgent(sess, goal, config)
        total_reward = []
        for step in range(config.num_episode):
            print('Epoch: ', step)
            vrep.simxFinish(-1)
            env = Env(config, goal)
            client = env.client
            step_reward = 0.0
            if client != -1:
                print("Connected to V-REP server")
                # state = normalize(env.state, env.max_error)
                state = env.state
                done = False
                while vrep.simxGetConnectionId(client) != -1:
                    if not done:
                        action = ddpg.noise_action(state)
                        print(ddpg.action(state))
                        reward, next_state, done = env.step(action)
                        # next_state = normalize(next_state, env.max_error)
                        # action = action / 10.0
                        ddpg.observe(state, action, reward, next_state, done)
                        state = next_state
                        step_reward += reward
                    else:
                        vrep.simxStopSimulation(client, vrep.simx_opmode_oneshot)
            else:
                "Couldn't connect to V-REP server!"
            vrep.simxFinish(client)
            total_reward.append(step_reward)
            print('Reward: ', step_reward)
            if step % 100 == 0:
                ddpg.save(step)


if __name__ == '__main__':
    os.system('clear')
    main()
