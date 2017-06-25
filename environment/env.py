import os
import time

from numpy import random

from vrep import vrep
from environment.robot import Robot
from environment.navigation import Navigation


class Env(object):

    def __init__(self, config, goal):
        self.config = config
        self.goal = goal

        self.prev_error = 0.0
        self.client = -1

    def reset(self, client):
        self.client = client
        self.robot, self.nav = self._spawn_robot(self.client)
        self.state = self._get_state()
        self.prev_error = self.state[5]
        return self.state

    def step(self, action):
        self.robot.set_motor_velocities(action)
        next_state = self._get_state()
        reward, done = self._reward(next_state)
        return reward, next_state, done

    def random_action(self):
        vel = []
        for i in range(2):
            vel.append(random.uniform(0.0, self.config.action_bound))
        self.robot.set_motor_velocities(vel)
        
    def _get_state(self):
        self.nav.compute_position(
            self.robot.read_encoders(), self.robot.read_gyroscope(),
            self.robot.position())
        dist = self.robot.read_proximity_sensors()
        error = self.nav.navigation_error.to_list()
        return dist + error

    def _reward(self, state):
        dist = state[0:-2]
        error = state[-2]
        reward = 0
        done = False
        if all(i < 0.03 for i in dist):
            done = True
            reward = -2.0
        elif all(i < 0.1 for i in dist):
            reward = -0.1
        else:
            reward = 2 * (self.prev_error - error)

        if error < 0.05:
            print('Target reached!')
            done = True
            reward = 10.0

        self.prev_error = error
        return reward, done

    def run_env(self):
        cmd = (self.config.vrep + self.config.vrep_param
               + self.config.path + self.config.scene)
        os.system(cmd)
        time.sleep(1)
        client = vrep.simxStart(
            '127.0.0.1', self.config.port, True, True, 5000, 5)
        return client

    def _spawn_robot(self, client):
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
            client, 0.06, 0.156, 0.1, 10.0, robot_streams, robot_objects)

        navigation = Navigation(
            robot.position(), self.goal, robot.wheel_diameter, robot.body_width)

        return robot, navigation
