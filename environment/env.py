import os
import time

import numpy as np

from vrep import vrep
from environment.robot import Robot
from environment.navigation import Navigation


class Env(object):

    def __init__(self, config, goal):
        self.config = config
        self.goal = goal

        self._client = self.run_env()
        if self._client != -1:
            print("Connected to V-REP")
            vrep.simxSynchronous(self._client, True)
            vrep.simxStartSimulation(
                self._client, vrep.simx_opmode_oneshot_wait)

            self._robot, self._nav = self._spawn_robot()
            self._state = self._get_state()

            self._prev_error = self._state[5]
            self._max_error = self._prev_error
            self.motion_check = [[], []]
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

    def step(self, action):
        t = time.time()
        action = (action * 10.0 + 10.0) / 2.0
        self._robot.set_motor_velocities(action)

        vrep.simxGetPingTime(self._client)
        vrep.simxSynchronousTrigger(self._client)

        next_state = self._get_state()
        reward, done = self._reward(next_state)
        print("Step time: ", time.time() - t)
        norm_next_state = self._normalize_state(next_state)
        return reward, norm_next_state, done

    def random_action(self):
        vel = np.zeros(2)
        for i in range(2):
            vel[i] = np.random.uniform(0.0, self.config.action_bound)
        self._robot.set_motor_velocities(vel)
        return vel

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
        print([dist, error])
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
            reward = 2 * (self._prev_error - error)

        if len(self.motion_check[0]) >= 300:
            if (np.abs(np.mean(self.motion_check[0])) > 0.2 or
                    np.abs(np.mean(self.motion_check[1])) < 0.002):
                reward = -2.0
                done = True
            else:
                self.motion_check = [[], []]
        else:
            self.motion_check[0].append(self._robot.read_gyroscope())
            self.motion_check[1].append(self._prev_error - error)

        if error < 0.05:
            print('Target reached!')
            done = True
            reward = 10.0

        self._prev_error = error
        reward /= (2 * self._max_error + 10.0)
        return np.round(reward, 6), done

    @staticmethod
    def run_env():
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
            robot.position(), self.goal, robot.wheel_diameter, robot.body_width)

        return robot, navigation

    def _normalize_state(self, state):
        state[0:-2] = state[0:-2] - 1.0
        state[-2] = (state[-2] - self._max_error / 2) / self._max_error / 2
        state[-1] = state[-1] / np.pi
        norm_state = np.clip(state, -1.0, 1.0)
        return norm_state
