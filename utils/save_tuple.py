import torch
import os

def save_tuple(state, next_state, action, reward, is_terminal, path, t):
    s_prime_aux = state
    s_prime_aux = s_prime_aux.detach().cpu().type(torch.uint8)
    s_aux = next_state
    
    s_aux = s_aux.detach().cpu().type(torch.uint8)
    
    action_aux = action
    action_aux = action_aux.detach().cpu().type(torch.uint8)
    reward_aux = reward
    reward_aux = torch.tensor([reward_aux]).type(torch.uint8).reshape(-1, 1)
    is_terminal_aux = is_terminal
    is_terminal_aux = torch.tensor([int(is_terminal_aux)]).type(torch.uint8).reshape(-1, 1)

    os.makedirs(path, exist_ok = True) 

    torch.save({
        "state" : s_aux,
        "next_state" : s_prime_aux,
        "action" : action_aux,
        "reward" : reward_aux,
        "is_terminal" : is_terminal_aux
        }, os.path.join(path, 'tuple_{}'.format(t)))