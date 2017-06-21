import numpy as np
import tensorflow as tf

from acotr import Actor
from critic import Critic
from robot import Robot
from navigation import Navigation

class DDPGAgent(object):

    def __init__(self, sess):
        self.sess = sess
        actor = Actor()
        critic = Critic()