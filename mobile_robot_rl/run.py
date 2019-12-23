import gym
import gym_vrep

import mobile_robot_rl.agents as agents
import mobile_robot_rl.networks.bodies as b
from mobile_robot_rl.common.env_wrapper import NavigationWrapper

if __name__ == '__main__':
    env = NavigationWrapper(gym.make('RoomNavigation-v0'))
    state_dim = 4 * env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent_sac = agents.SAC(
        pi_phi=b.MLPNetwork(state_dim, (512, 256)),
        qv_phi=b.MLPNetwork((state_dim + action_dim), (512, 256)),
        env=env,
        state_dim=state_dim,
        discount_factor=0.99,
        memory_capacity=int(1e6),
        batch_size=256,
        warm_up_steps=10000,
        update_steps=1,
        use_soft_update=True,
        pi_lrate=5e-4,
        qv_lrate=5e-4,
        alpha_lrate=5e-4,
        n_step=3

    )
    agent_sac.train(1000000)
