import os
import time

import numpy as np

from vrep import vrep
from environment.robot import Robot
from environment.navigation import Navigation


class Env(object):

    def __init__(self, goal, viz=False):
        self._goal = goal

        self._client = self._run_env(viz)
        if self._client != -1:
            print("Connected to V-REP")
            vrep.simxSynchronous(self._client, True)
            vrep.simxStartSimulation(
                self._client, vrep.simx_opmode_oneshot_wait)

            self._robot, self._nav = self._spawn_robot()
            self._state = self._get_state()

            self._prev_error = self._state[5]
            self._max_error = self._prev_error
            self._state_c = (self._max_error - np.pi) / 2.0
            self._state_ac = (self._max_error + np.pi) / 2.0
            self._motion_check = []
        else:
            print("Couldn't connect to V-REP!")
            os.system("pkill vrep")

    @property
    def client(self):
        return self._client

    @property
    def state(self):
        return self._state

    @property
    def max_error(self):
        return self._max_error

    @property
    def current_error(self):
        return self._nav.navigation_error[0]

    @property
    def path(self):
        return self._nav.path

    def step(self, norm_action):
        self._robot.set_motor_velocities(norm_action)

        vrep.simxGetPingTime(self._client)
        vrep.simxSynchronousTrigger(self._client)

        next_state = self._get_state()
        reward, done = self._reward(next_state)
        norm_next_state = self._normalize_state(next_state)
        return reward, norm_next_state, done

    def _get_state(self):
        self._nav.compute_position(
            self._robot.read_encoders(), self._robot.read_gyroscope(),
            self._robot.position())
        dist = self._robot.read_proximity_sensors()
        if dist.size == 0.0:
            dist = self.dist
        else:
            self.dist = dist
        error = self._nav.navigation_error

        return np.concatenate((dist, error))

    def _reward(self, state):
        dist = state[0:5]
        error = state[5:7]
        done = False
        if not all(i > 0.04 for i in dist):
            done = True
            reward = -1.0
        else:
            reward = (self._prev_error - error[0]) * 30

        if len(self._motion_check) >= 300:
            if np.abs(np.mean(self._motion_check)) < 0.001:
                reward = -1.0
                done = True
            else:
                self._motion_check = []
        else:
            self._motion_check.append(self._prev_error - error)

        if error[0] < 0.05:
            print('Target reached!')
            done = True
            reward = 1.0

        self._prev_error = error[0]
        return np.round(np.clip(reward, -1.0, 1.0), 3), done

    @staticmethod
    def _run_env(viz):
        scene = "navigation_task_0.ttt &"
        cmd_2 = "vrep.sh -h -q "
        cmd_3 = "-gREMOTEAPISERVERSERVICE_20000_FALSE_TRUE "
        cmd_4 = "/home/souphis/Magisterka/Simulation/" + scene
        os.system(cmd_2 + cmd_3 + cmd_4)
        time.sleep(0.5)
        client = vrep.simxStart(
            '127.0.0.1', 20000, True, True, 5000, 5)
        return client

    def _spawn_robot(self):
        vrep.simxFinish(-1)
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
            self._client, 0.06, 0.156, 0.05, 10.0, robot_streams, robot_objects)
        vrep.simxGetPingTime(self._client)
        vrep.simxSynchronousTrigger(self._client)
        for i in range(5):
            robot.position()
            vrep.simxSynchronousTrigger(self._client)
        navigation = Navigation(
            robot.position(), self._goal, robot.wheel_diameter,
            robot.body_width)

        return robot, navigation

    def _normalize_state(self, norm_state):
        return np.clip((norm_state - self._state_c) / self._state_ac, -1.0, 1.0)
