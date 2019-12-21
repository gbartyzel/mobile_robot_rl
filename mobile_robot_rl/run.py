import copy
import numpy as np
import gym
import gym_vrep

import mobile_robot_rl.agents as agents
import mobile_robot_rl.networks.bodies as b
from mobile_robot_rl.common.env_wrapper import NavigationWrapper


if __name__ == '__main__':
    env = NavigationWrapper(gym.make('RoomNavigation-v0'))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent_sac = agents.SAC(
        pi_phi=b.BNMLPNetwork(state_dim, (256, 256)),
        qv_phi=b.BNMLPNetwork((state_dim + action_dim), (256, 256)),
        env=env,
        discount_factor=0.99,
        memory_capacity=int(1e6),
        batch_size=256,
        warm_up_steps=10000,
        update_steps=1,
        use_soft_update=True,
        pi_lrate=5e-4,
        qv_lrate=5e-4,
        alpha_lrate=5e-4

    )
    agent_sac.train(1000000)
