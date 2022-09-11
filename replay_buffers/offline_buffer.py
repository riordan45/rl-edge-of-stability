import torch
from collections import namedtuple
from replay_buffers.base_buffer import base_buffer

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')

class offline_replay_buffer(base_buffer):
    def __init__(self, path, first, last):
        super().__init__()
        
        self.buffer = []
        for i in range(first, last):
            # print(i)
            aux = torch.load(path + "/tuple_{}".format(i))

            self.buffer.append(transition(aux['state'], aux['next_state'], aux['action'], aux['reward'], aux['is_terminal']))
