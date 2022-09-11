import torch
import random
from replay_buffers import offline_replay_buffer
from collections import namedtuple

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')

from os import listdir
from os.path import isfile, join

import sys
if __name__ == "__main__":
    last = 4990000
    load_idx = random.sample(list(range(5000000)), 500000)
    
    path = "/home/freshpate/offline_rl_data/minatar/breakout_torch_all_replay_adam"
    # onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # print(len(onlyfiles))
    # sys.exit()?
    # load_idx = list(range(10000))
    to_buffer = []
    for i in load_idx:
        aux = torch.load(path + "/tuple_{}".format(i))
        to_buffer.append(transition(aux['state'], aux['next_state'], aux['action'], aux['reward'], aux['is_terminal']))
    r_buffer = offline_replay_buffer("/home/freshpate/offline_rl_data/minatar/breakout_torch_all_replay_adam", 0, 1)
    r_buffer.buffer = to_buffer
    torch.save({"buffer" : r_buffer}, "/home/freshpate/offline_rl_data/minatar/breakout_last10percent")