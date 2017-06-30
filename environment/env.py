import os
import time
import random as rnd

import numpy as np

from vrep import vrep
from environment.robot import Robot
from environment.navigation import Navigation


class Env(object):

    def __init__(self, config, goal):
        self.config = config
        self.goal = goal

        self._client = self.run_env()
        self.robot, self.nav = self._spawn_robot()
        self._state = self._get_state()

        self.prev_error = self._state[5]
        self._max_error = self.prev_error
        self.motion_check = [[], []]

    @property
    def client(self):
        return self._client

    @property
    def state(self):
        return self._state

    @property
    def max_error(self):
        return self._max_error

    def step(self, action):
        self.robot.set_motor_velocities(action)
        next_state = self._get_state()
        reward, done = self._reward(next_state)
        return reward, next_state, done

    def random_action(self):
        vel = np.zeros(2)
        for i in range(2):
            vel[i] = np.random.uniform(0.0, self.config.action_bound)
        self.robot.set_motor_velocities(vel)
        return vel

    def _get_state(self):
        self.nav.compute_position(
            self.robot.read_encoders(), self.robot.read_gyroscope(),
            self.robot.position())
        dist = self.robot.read_proximity_sensors()
        error = self.nav.navigation_error

        return np.concatenate((dist, error))

    def _reward(self, state):
        dist = state[0:-2]
        error = state[-2]
        done = False

        if not all(i > 0.03 for i in dist):
            done = True
            reward = -2.0
        elif not all(i > 0.1 for i in dist):
            reward = -0.1
        else:
            reward = 2 * (self.prev_error - error)

        if len(self.motion_check[0]) >= 300:
            if (np.abs(np.mean(self.motion_check[0])) > 0.1 or
                    np.abs(np.mean(self.motion_check[1])) < 0.01):
                reward -= 2.0
                done = True
            else:
                self.motion_check = [[], []]
        else:
            self.motion_check[0].append(self.robot.read_gyroscope())
            self.motion_check[1].append(self.robot.read_velocities()[2])

        if error < 0.05:
            print('Target reached!')
            done = True
            reward = 10.0

        self.prev_error = error
        return reward, done

    @staticmethod
    def run_env():
        scene = "s_navigation_task_0.ttt &"
        cmd_2 = "vrep.sh -s120000 -q "
        cmd_3 = "-gREMOTEAPISERVERSERVICE_20000_FALSE_FALSE "
        cmd_4 = "/home/souphis/Magisterka/Simulation/" + scene
        os.system(cmd_2 + cmd_3 + cmd_4)
        time.sleep(10)
        client = vrep.simxStart(
            '127.0.0.1', 20000, True, True, 5000, 5)
        return client

    def _spawn_robot(self):
        robot_objects = (
            'smartBotLeftMotor',
            'smartBotRightMotor',
            'smartBot'
        )

        robot_streams = (
            'distanceSignal',
            'accelSignal',
            'gyroSignal',
            'leftEncoder',
            'rightEncoder'
        )

        robot = Robot(
            self._client, 0.06, 0.156, 0.1, 10.0, robot_streams, robot_objects)
        [robot.position() for i in range(2000)]
        navigation = Navigation(
            robot.position(), self.goal, robot.wheel_diameter, robot.body_width)

        return robot, navigation
