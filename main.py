import os
import time
import numpy as np
import tensorflow as tf

from ddpg import DDPGAgent
from environment.env import Env
from utils.config import AgentConfig
from vrep import vrep


def main():
    goal = (2.0, 2.0)
    config = AgentConfig
    with tf.Session() as sess:
        ddpg = DDPGAgent(sess, goal, config)
        total_reward = []
        total_steps = 0.0
        for ep in range(config.num_episode):
            print('Epoch: ', ep)
            vrep.simxFinish(-1)
            env = Env(config, goal)
            client = env.client
            step_reward = 0.0
            if client != -1:
                print("Connected to V-REP server")
                state = env.state
                vrep.simxSynchronousTrigger(client)
                done = False
                step = 0
                while step < 2400:
                    if not done:
                        action = ddpg.noise_action(state)
                        print(action)
                        reward, next_state, done = env.step(action)
                        t = time.time()
                        ddpg.observe(state, action, reward, next_state, done)
                        print("Train time ", time.time() - t)
                        state = next_state
                        step_reward += reward
                        step += 1
                    else:
                        vrep.simxStopSimulation(client, vrep.simx_opmode_oneshot)
                        if vrep.simxGetConnectionId(client) == -1:
                            break
                total_steps += step
            else:
                "Couldn't connect to V-REP server!"
            vrep.simxFinish(client)
            total_reward.append(step_reward)
            print('Reward: ', step_reward)
            if ep % 100 == 0:
                ddpg.save(ep)


if __name__ == '__main__':
    tf.reset_default_graph()
    os.system('clear')
    main()
