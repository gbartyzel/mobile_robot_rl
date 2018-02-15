import random
from collections import deque, namedtuple


Transition = namedtuple('Transition', 'state action reward next_state terminal')


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity

        self.buffer = deque()
        self._size = 0

    def add(self, state, action, reward, next_state, done):
        self._size += 1
        experience = Transition(state, action, reward, next_state, done)

        if self.size >= self.capacity:
            self.buffer.popleft()
            self.buffer.append(experience)
        else:
            self.buffer.append(experience)

        self._size = min(self.capacity, self._size)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    @property
    def size(self):
        return self._size
