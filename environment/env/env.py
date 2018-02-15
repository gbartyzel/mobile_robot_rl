import os
import time

import numpy as np

from environment.navigation.ideal import Ideal
from environment.navigation.odometry import Odometry
from environment.navigation.gyrodometry import Gyrodometry
from vrep import vrep


class Env(object):
    """Docstring for Env. """

    def __init__(self,
                 env_name,
                 navigation_method='ideal',
                 visulalization=False):
        """TODO: to be defined1.

        :env_name: TODO
        :visulalization: TODO

        """
        self._env_name = env_name
        self._navigation_method = navigation_method
        self._visulalization = visulalization

        self._envs = {
            "no_obstacles": ("no_obstacles.ttt",
                             np.round((np.random.rand(2) - 1) * 2, 3)),
            "room": ("room.ttt", np.array([2.0, 2.0])),
            "room_random":
            ["room_1.ttt", "room_2.ttt", "room_3.ttt", "room_4.ttt"]
        }

    def reset(self):
        self._client = self.run_env()
        if self._client != -1:
            print("Connected to V-REP")
            vrep.simxSynchronous(self._client, True)
            vrep.simxStartSimulation(self._client,
                                     vrep.simx_opmode_oneshot_wait)

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

    def run_env(self):
        if self._visulalization:
            vrep_exec = os.environ["V_REP"] + "/vrep.sh -q "
        else:
            vrep_exec = os.environ["V_REP"] + "/vrep.sh -h -q "
        cmd = "/opt/V-REP/vrep.sh -h -q " \
              "-gREMOTEAPISERVERSERVICE_20000_FALSE_TRUE " \
              "simulations/" + scene
        os.system(cmd)
        time.sleep(0.5)
        return vrep.simxStart('127.0.0.1', 20000, True, True, 5000, 5)
