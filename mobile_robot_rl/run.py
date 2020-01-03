import copy

import gym
import gym_vrep
import numpy as np

import mobile_robot_rl.agents as agents
import mobile_robot_rl.networks.bodies as b
from mobile_robot_rl.common.env_wrapper import NavigationWrapper
from mobile_robot_rl.common.history import FusionHistory
from mobile_robot_rl.common.memory import Dimension

if __name__ == '__main__':
    env = NavigationWrapper(gym.make('DynamicRoomVisionNavigation-v0'))
    action_dim = env.action_space.shape[0]

    state_dim = dict(
        scalars=Dimension(4 * env.observation_space['scalars'].shape[0],
                          np.float32),
        image=Dimension((4, 64, 64), np.uint8))

    history = FusionHistory(4, (64, 64), 14, True, 'scalars', 'image')
    fusion_model = b.FusionModel(state_dim['scalars'].shape,
                                 state_dim['image'].shape[0], (512, 256))
    agent_sac = agents.SAC(
        pi_phi=copy.deepcopy(fusion_model),
        qv_phi=b.CriticFusionModel(action_dim, (256,),
                                   copy.deepcopy(fusion_model)),
        env=env,
        state_dim=state_dim,
        discount_factor=0.99,
        memory_capacity=int(1e5),
        batch_size=128,
        warm_up_steps=10000,
        update_steps=1,
        use_soft_update=True,
        pi_lrate=5e-4,
        qv_lrate=5e-4,
        alpha_lrate=5e-4,
        n_step=1,
        history=history
    )
    agent_sac.train(1000000)
