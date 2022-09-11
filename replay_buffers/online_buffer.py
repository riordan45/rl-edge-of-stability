from collections import namedtuple
from replay_buffers.base_buffer import base_buffer

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')

class online_replay_buffer(base_buffer):
    def __init__(self, buffer_size):
        super().__init__()

        self.buffer_size = buffer_size
        self.location = 0

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size