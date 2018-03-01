import os
import subprocess
import time

import numpy as np

from environment.navigation.ideal import Ideal
from environment.navigation.odometry import Odometry
from environment.navigation.gyrodometry import Gyrodometry
from environment.robot import Robot
from environment.scenes import ENV_SCENES

from environment.vrep import vrep


class Env(object):
    def __init__(self,
                 env_name,
                 navigation_method="ideal",
                 visulalization=False,
                 normalization=False):

        self._scene = ENV_SCENES[env_name]["scene_file"]
        self._max_steps = ENV_SCENES[env_name]["steps"]
        self._target_position = ENV_SCENES[env_name]["target_position"]
        self._dt = ENV_SCENES[env_name]["dt"]
        self._action_dim = ENV_SCENES[env_name]["action_dim"]
        self._observation_dim = ENV_SCENES[env_name]["observation_dim"]
        self._robot_model = ENV_SCENES[env_name]["robot_model"]

        self._navigation_method = navigation_method
        self._normalization = normalization
        self._visulalization = visulalization

        self._client = None
        self._robot = None
        self._nav = None

        self._prev_error = 0.0
        self._max_error = 0.0
        self._motion_check = []

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def scene(self):
        return self._scene

    @property
    def target_position(self):
        return self._target_position

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def dt(self):
        return self._dt

    def reset(self):
        vrep.simxFinish(-1)
        self._client = self._run_env()
        if self._client != -1:
            print("Connected to V-REP")
            vrep.simxSynchronous(self._client, True)
            vrep.simxStartSimulation(self._client, vrep.simx_opmode_oneshot)

            self._robot, self._nav = self._spawn_robot()
            state = self._get_state()

            self._prev_error = state[5]
            self._max_error = self._prev_error
            self._motion_check = []

            return state
        else:
            subprocess.call("pkill vrep &", shell=True)
            print("Couldn't connect to V-REP!")

    def step(self, action):
        if self._normalization:
            action = self._rescale_action(action)
        self._robot.set_motor_velocities(action)

        vrep.simxSynchronousTrigger(self._client)

        next_state = self._get_state()
        reward, done = self._reward(next_state)
        if self._normalization:
            next_state = self._normalize_state(next_state)

        return reward, next_state, done

    def stop(self):
        vrep.simxStopSimulation(self._client, vrep.simx_opmode_oneshot)
        while vrep.simxGetConnectionId(self._client) != -1:
            vrep.simxSynchronousTrigger(self._client)

    def _run_env(self):
        if self._visulalization:
            vrep_exec = os.environ["V_REP"] + "vrep.sh -q "
        else:
            vrep_exec = os.environ["V_REP"] + "vrep.sh -h -q "
        synch_mode_cmd = "-gREMOTEAPISERVERSERVICE_20000_FALSE_TRUE "
        scene = "./environment/scenes/" + self._scene

        subprocess.call(vrep_exec + synch_mode_cmd + scene + " &", shell=True)
        time.sleep(5.0)
        return vrep.simxStart("127.0.0.1", 20000, True, True, 5000, 5)

    def _spawn_robot(self):
        robot = Robot(self._client, self._robot_model["robot_streams"],
                      self._robot_model["robot_objects"],
                      self._robot_model["wheel_diameter"],
                      self._robot_model["body_width"], self._dt,
                      self._robot_model["min_velocity"],
                      self._robot_model["max_velocity"])

        vrep.simxGetPingTime(self._client)
        vrep.simxSynchronousTrigger(self._client)

        for _ in range(5):
            robot.get_position()
            vrep.simxSynchronousTrigger(self._client)

        if self._navigation_method == "ideal":
            navigation = Ideal(self._target_position,
                               self._robot_model["wheel_diameter"],
                               self._robot_model["body_width"], self._dt)
        elif self._navigation_method == "odometry":
            navigation = Odometry(robot.get_position(), self._target_position,
                                  self._robot_model["wheel_diameter"],
                                  self._robot_model["body_width"], self._dt)
        elif self._navigation_method == "gyrodometry":
            navigation = Gyrodometry(robot.get_position(),
                                     self._target_position,
                                     self._robot_model["wheel_diameter"],
                                     self._robot_model["body_width"], self._dt)
        else:
            raise ValueError("Invalid nevigation method")

        return robot, navigation

    def _get_state(self):
        self._nav.compute_position(
            position=self._robot.get_position(),
            phi=self._robot.get_encoders_values(),
            angular_velocity=self._robot.get_gyroscope_values())

        dist = self._robot.get_proximity_values()
        while len(dist) == 0:
            dist = self._robot.get_proximity_values()
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
            reward = 30 * (self._prev_error - error[0])

        if len(self._motion_check) >= 300:
            if np.abs(np.mean(self._motion_check)) < 0.001:
                reward = -1.0
                done = True
            else:
                self._motion_check = []
        else:
            self._motion_check.append(self._prev_error - error[0])

        if error[0] < 0.05:
            print('Target reached!')
            done = True
            reward = 1.0

        self._prev_error = error[0]
        return np.clip(reward, -1.0, 1.0), done

    def _rescale_action(self, norm_action):
        rescaled_action = (
            ((norm_action + 1.0) *
             (self._robot.vel_bound[1] + self._robot.vel_bound[0])) / 2.0 +
            self._robot.vel_bound[0])
        return rescaled_action

    def _normalize_state(self, norm_state):
        state_c = (self._max_error - np.pi) / 2.0
        state_ac = (self._max_error + np.pi) / 2.0
        return np.clip((norm_state - state_c) / state_ac, -1.0, 1.0)


if __name__ == "__main___":
    env = Env("room")
