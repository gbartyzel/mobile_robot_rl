import copy

import gym
import gym_vrep
import numpy as np

import mobile_robot_rl.agents as agents
import mobile_robot_rl.networks.bodies as b
from mobile_robot_rl.common.env_wrapper import NavigationWrapper
from mobile_robot_rl.common.history import FusionHistory, VectorHistory
from mobile_robot_rl.common.memory import Dimension


def run_vision_env():
    env = NavigationWrapper(gym.make('DynamicRoomVisionNavigation-v0'))
    action_dim = env.action_space.shape[0]

    state_dim = dict(
        scalars=Dimension((4, env.observation_space['scalars'].shape[0]),
                          np.float32),
        image=Dimension((4, 64, 64), np.uint8))

    history = FusionHistory(4, (64, 64), 14, False, 'scalars', 'image')
    fusion_model = b.FusionModel(state_dim['image'].shape[0], (512, ))

    agent_sac = agents.SAC(
        pi_phi=copy.deepcopy(fusion_model),
        qv_phi=b.CriticFusionModel(action_dim, (512,),
                                   copy.deepcopy(fusion_model)),
        env=env,
        state_dim=state_dim,
        discount_factor=0.99,
        memory_capacity=int(1e5),
        batch_size=64,
        warm_up_steps=10000,
        update_steps=4,
        use_soft_update=True,
        pi_lrate=5e-4,
        qv_lrate=5e-4,
        alpha_lrate=5e-4,
        n_step=1,
        history=history,
        polyak_factor=0.005
    )
    agent_sac.train(1000000)


def run_env():
    env = NavigationWrapper(gym.make('DynamicRoomNavigation-v0'))
    action_dim = env.action_space.shape[0]

    state_dim = Dimension(4 * env.observation_space.shape[0],
                          np.float32)

    history = VectorHistory(4, env.observation_space.shape[0], True)

    agent_sac = agents.SAC(
        pi_phi=b.MLPNetwork(state_dim.shape, (256, 256)),
        qv_phi=b.MLPNetwork((state_dim.shape + action_dim), (256, 256)),
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
        n_step=3,
        history=history
    )
    agent_sac.train(1000000)


if __name__ == '__main__':
    run_vision_env()
