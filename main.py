import argparse
from itertools import count
import logging
import os
import time
import torch
import random
import utils
import numpy as np
from collections import namedtuple
from compute_eigs import get_hessian_eigenvalues
from replay_buffers import offline_replay_buffer, online_replay_buffer
from agents import CategoricalAgent, DQNAgent
from minatar import Environment
from prettytable import PrettyTable
transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def world_dynamics(t, state, env, agent, cmdl):

    # A uniform random policy is run before the learning starts
    if cmdl.online:
        if t < cmdl.replay_start_size:
            action = torch.tensor([[random.randrange(agent.no_actions)]], device=agent.device)
        else:
            # Epsilon-greedy behavior policy for action selection
            # Epsilon is annealed linearly from 1.0 to END_EPSILON over the cmdl.decay_epsilon_for_these_timesteps and stays 0.1 for the
            # remaining frames
            epsilon = cmdl.end_epsilon if t - cmdl.replay_start_size >= cmdl.decay_epsilon_for_these_timesteps \
                else ((cmdl.end_epsilon - cmdl.start_epsilon) / cmdl.decay_epsilon_for_these_timesteps) * (t - cmdl.replay_start_size) + cmdl.start_epsilon

            if np.random.binomial(1, epsilon) == 1:
                action = torch.tensor([[random.randrange(agent.no_actions)]], device=agent.device)
            else:
                # State is 10x10xchannel, max(1)[1] gives the max action value (i.e., max_{a} Q(s, a)).
                # view(1,1) shapes the tensor to be the right form (e.g. tensor([[0]])) without copying the
                # underlying tensor.  torch._no_grad() avoids tracking history in autograd.
                action = agent.evaluate(state)[1].view(1, 1)
    else:
        action = agent.evaluate(state)[1].view(1, 1)
    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime
    s_prime = agent.get_minatar_state(env.state())

    return s_prime, action, torch.tensor([[reward]], device=agent.device).float(), torch.tensor([[terminated]], device=agent.device)



def train(env, cmdl):

    # Get channels and number of actions specific to each game
    in_channels = env.state_shape()[2]
    no_actions = env.num_actions()

    if cmdl.agent == "dqn":
        agent = DQNAgent(input_channels=in_channels, 
                         no_actions=no_actions, 
                         gamma=cmdl.gamma, 
                         batch_size=cmdl.batch_size)
    
    elif cmdl.agent == "c51":
        agent = CategoricalAgent(input_channels=in_channels, 
                                no_actions=no_actions, 
                                gamma=cmdl.gamma, 
                                batch_size=cmdl.batch_size,
                                lanczos_batch_size=cmdl.batch_size,
                                atoms=cmdl.atoms, 
                                v_min=cmdl.v_min, 
                                v_max=cmdl.v_max)
    
    # Instantiate networks, optimizer, loss and buffer
    if cmdl.online:
        r_buffer = online_replay_buffer(cmdl.replay_buffer_size)
        # prime_r_buffer = torch.load(cmdl.tuple_data_path)['buffer']
    else:
        # r_buffer = offline_replay_buffer(path=cmdl.tuple_data_path, first=0, last=1000000)
        # print("saving")
        r_buffer = torch.load(cmdl.tuple_data_path)['buffer']
        # new_buff = offline_replay_buffer("/home/freshpate/offline_rl_data/minatar/breakout_torch_all_replay_adam", first=0, last=1)
        # new_buff.buffer = r_buffer.buffer
        # torch.save({"buffer" : r_buffer}, "/home/freshpate/offline_rl_data/minatar/space_invaders_offline_000_buff")
        # print("saving done")
        # import sys
        # sys.exit()
        # print(len(r_buffer.buffer))
    

    if cmdl.optimizer == "adamw":
        # output_path = os.path.join(cmdl.checkpoint_path, 
        #                            cmdl.agent, 
        #                            training_regime,
        #                            cmdl.optimizer, 
        #                            str(cmdl.lr), 
        #                            str(cmdl.beta1), 
        #                            str(cmdl.beta2), 
        #                            str(cmdl.weight_decay))

        optimizer = torch.optim.AdamW(agent.policy_net.parameters(), 
                                      lr=cmdl.lr, 
                                      betas=(cmdl.beta1, cmdl.beta2), 
                                      eps=cmdl.adam_eps / cmdl.batch_size,
                                      weight_decay=cmdl.weight_decay)

    elif cmdl.optimizer == "adam":
        # output_path = os.path.join(cmdl.checkpoint_path, 
        #                            cmdl.agent,
        #                            training_regime,
        #                            cmdl.optimizer, 
        #                            str(cmdl.lr), 
        #                            str(cmdl.beta1), 
        #                            str(cmdl.beta2))

        optimizer = torch.optim.Adam(agent.policy_net.parameters())
    
    elif cmdl.optimizer == "sgd":
        # output_path = os.path.join(cmdl.checkpoint_path, 
        #                            cmdl.agent,
        #                            training_regime,
        #                            cmdl.optimizer,  
        #                            str(cmdl.lr), 
        #                            str(cmdl.momentum))

        optimizer = torch.optim.SGD(agent.policy_net.parameters(), 
                                    lr=cmdl.lr,
                                    momentum=cmdl.momentum,
                                    nesterov=cmdl.nesterov)

    # Set initial values
    count_parameters(agent.policy_net)
    print(cmdl.optimizer)
    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0
    avg_return_init = 0.0
    data_return_init = []
    frame_stamp_init = []
    load_path = cmdl.load_checkpoint_path
    # Load model and optimizer if load_path is not None
    if load_path is not None and isinstance(load_path, str):
        checkpoint = torch.load(load_path)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        r_buffer = checkpoint['replay_buffer']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e_init = checkpoint['episode']
        t_init = checkpoint['frame']
        policy_net_update_counter_init = checkpoint['policy_net_update_counter']
        avg_return_init = checkpoint['avg_return']
        data_return_init = checkpoint['return_per_run']
        frame_stamp_init = checkpoint['frame_stamp_per_run']

        # Set to training mode
        agent.policy_net.train()
        agent.old_policy_net.train()
        if not cmdl.target_off:
            agent.target_net.train()

    agent.old_policy_net.train()
    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init

    # Train for a number of frames
    t = t_init
    e = e_init
    total_grad_norm = []
    total_loss = []
    total_update_norm = []
    total_eigs = []
    total_scalar_product = []
    policy_net_update_counter = policy_net_update_counter_init

    ##### Please specify here online, offline, corrupted_offline_percentage, hybrid offline from online buffer #####
    training_regime = "last_10k"
    print("Current training regime is: {}".format(training_regime))

    output_path = os.path.join(cmdl.checkpoint_path, cmdl.agent)
    key = "{}_{}_{}_{}_{}_{}".format(cmdl.game,
                                     cmdl.agent,
                                     training_regime,
                                     agent.batch_size,
                                     cmdl.optimizer,
                                     agent.date_and_time)
    os.makedirs(output_path, exist_ok = True)
    
    t_start = time.time()
    prime_sample = r_buffer.sample(cmdl.batch_size)
    eigs, product = get_hessian_eigenvalues(agent, prime_sample, cmdl.batch_size, neigs=cmdl.no_eigs)
                        # total_eigs.append(eigs)
                        # total_scalar_product.append(product)
    while t < cmdl.training_steps:
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0
        # if t == 200000:
        #     for g in optimizer.param_groups:
        #             g['lr'] = g['lr'] / 2

        # Initialize the environment and start state
        env.reset()
        s = agent.get_minatar_state(env.state())
        is_terminated = False
        while(not is_terminated) and t < cmdl.training_steps:
            # Generate data
            s_prime, action, reward, is_terminated = world_dynamics(t, s, env, agent, cmdl)
            sample = None
            # if t == 200000:
            #     for g in optimizer.param_groups:
            #         g['lr'] = g['lr'] / 1.5
            
            if cmdl.online:
                
                # Write the current frame to replay buffer
                r_buffer.add(s, s_prime, action, reward, is_terminated)
                if cmdl.save_online_replay:
                    r_buffer.save(s, s_prime, action, reward, is_terminated, t, cmdl.tuple_data_path)

                if t > cmdl.replay_start_size and len(r_buffer.buffer) >= agent.batch_size:
                    # Sample a batch
                    sample = r_buffer.sample(agent.batch_size)
            else:
                sample = r_buffer.sample(agent.batch_size)

            # Train every n number of frames defined by cmdl.train_freq
            if t % cmdl.train_freq == 0 and sample is not None:
                policy_net_update_counter += 1
                
                optimizer.zero_grad()
                loss = agent.train(agent.batch_size, *agent.process_sample(sample))
                loss.backward()
                optimizer.step()
                if t % cmdl.logging_frequency == 0:
                    if cmdl.compute_eigs:
                        eigs, product = get_hessian_eigenvalues(agent, prime_sample, cmdl.batch_size, neigs=cmdl.no_eigs)
                        total_eigs.append(eigs)
                        total_scalar_product.append(product)
                        # print(eigs)

                if cmdl.compute_grad_norm:
                    norm = agent.calculate_gradient_norm(agent.policy_net)
                    total_grad_norm.append(norm)
                    # print("Grad norm is: {}".format(norm))
                
                if cmdl.compute_update_norm:
                    norm = agent.calculate_update_norm()
                    total_update_norm.append(norm)
                    # print("Update norm is: {}".format(norm))

                total_loss.append(loss.item())

            # # Update the target network only after some number of policy network updates
            if policy_net_update_counter > 0 and policy_net_update_counter % cmdl.target_network_update_freq == 0:
                agent.update_target_network(agent.policy_net)

            G += reward.item()

            t += 1

            # Continue the process
            s = s_prime

        # Increment the episodes
        e += 1

        # print("Episode : {} | Return : {}".format(e, G))
        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        if e % cmdl.logging_frequency_episode == 0:
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(np.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t)
                         + " | Eigenvalues: " + str(eigs))

        # Save model data and other intermediate data if the corresponding flag is true
        if cmdl.save_checkpoint and e % cmdl.logging_frequency_episode == 0 and sample is not None:
            
            torch.save({
                        'episode': e,
                        'frame': t,
                        'policy_net_update_counter': policy_net_update_counter,
                        'policy_net_state_dict': agent.policy_net.state_dict(),
                        'target_net_state_dict': agent.target_net.state_dict() if not cmdl.target_off else [],
                        'optimizer_state_dict': optimizer.state_dict(),
                        'avg_return': avg_return,
                        'return_per_run': data_return,
                        'frame_stamp_per_run': frame_stamp,
                        'grad_norm' : total_grad_norm,
                        'eigs' : total_eigs,
                        'loss' : total_loss,
                        'update_norm' : total_update_norm,
                        'batch_size' : agent.batch_size,
                        'training_regime' : training_regime,
                        'principal_scalar_product' : total_scalar_product
            }, os.path.join(output_path, key))
            

    logging.info("Avg return: " + str(np.around(avg_return, 2)) + " | Time per frame: " + str((time.time()-t_start)/t))
        
    # Write data to file
    torch.save({
                'episode': e,
                'frame': t,
                'policy_net_update_counter': policy_net_update_counter,
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict() if not cmdl.target_off else [],
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_return': avg_return,
                'return_per_run': data_return,
                'frame_stamp_per_run': frame_stamp,
                'grad_norm' : total_grad_norm,
                'eigs' : total_eigs,
                'loss' : total_loss,
                'update_norm' : total_update_norm,
                'batch_size' : agent.batch_size,
                'training_regime' : training_regime,
                'principal_scalar_product' : total_scalar_product
            }, os.path.join(output_path, key))

def main():
    cmdl = utils.get_config()
    if cmdl.verbose:
        logging.basicConfig(level=logging.INFO)

    # If there's an output specified, then use the user specified output.  Otherwise, create file in the current
    # directory with the game's name.

    env = Environment(cmdl.game)

    print('Cuda available?: ' + str(torch.cuda.is_available()))
    print('Remember to specify training regime')

    train(env, cmdl)

if __name__ == "__main__":
    main()