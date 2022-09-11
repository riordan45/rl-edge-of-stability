import os
import random
import torch

class base_buffer:
    def __init__(self):
        self.buffer = []

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def save(self, s, s_prime, action, reward, is_terminated, t, path):
        s_prime_aux = s_prime
        s_prime_aux = s_prime_aux.detach().cpu().type(torch.uint8)
        s_aux = s
        
        s_aux = s_aux.detach().cpu().type(torch.uint8)
        
        action_aux = action
        action_aux = action_aux.detach().cpu().type(torch.uint8)
        reward_aux = reward
        reward_aux = torch.tensor([reward_aux]).type(torch.uint8).reshape(-1, 1)
        is_terminal_aux = is_terminated
        is_terminal_aux = torch.tensor([int(is_terminal_aux)]).type(torch.uint8).reshape(-1, 1)

        output_path = os.path.join(path, "tuple_{}".format(t))
        torch.save({
            "state" : s_aux,
            "next_state" : s_prime_aux,
            "action" : action_aux,
            "reward" : reward_aux,
            "is_terminal" : is_terminal_aux
            }, output_path)