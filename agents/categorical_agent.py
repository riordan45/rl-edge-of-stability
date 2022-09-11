
import abc
import torch
from agents.base_agent import BaseAgent
from networks import CategoricalQNetwork
from collections import namedtuple

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')

class CategoricalAgent(BaseAgent, abc.ABC):
    def __init__(self, input_channels, no_actions, gamma, batch_size, lanczos_batch_size, atoms, v_min, v_max):
        super().__init__(input_channels=input_channels, no_actions=no_actions, gamma=gamma)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = CategoricalQNetwork(self.input_channels, no_actions, atoms, 128)
        self.old_policy_net = CategoricalQNetwork(self.input_channels, no_actions, atoms, 128)
        self.target_net = CategoricalQNetwork(self.input_channels, no_actions, atoms, 128)
        self.atoms = atoms
        self.support = torch.linspace(v_min, v_max, self.atoms, device=self.device)
        self.v_min = v_min
        self.v_max = v_max
        self.batch_size = batch_size
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.m = torch.zeros(batch_size, self.atoms, device=self.device).float()
        self.m_lanczos = torch.zeros(lanczos_batch_size, self.atoms, device=self.device).float()

        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.old_policy_net.to(self.device)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

    def calculate_update_norm(self):
        
        total_norm = 0
        new_parameters = [p for p in self.policy_net.parameters() if p.grad is not None and p.requires_grad]
        old_paremeters = [p for p in self.old_policy_net.parameters()]
        # print("intrat pe aici", old_paremeters, new_parameters)
        for old, new in zip(old_paremeters, new_parameters):
            param_norm = (new.detach().data - old.detach().data).norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

        return total_norm

    def evaluate(self, state):

        with torch.no_grad():
            probs = self.policy_net(state)
            support = self.support.expand_as(probs)
            q_val, argmax_a = torch.mul(probs, support).squeeze().sum(1).max(0)
            return (q_val, argmax_a)

    def train(self, batch_size, states, next_states, actions, rewards, is_terminal):
        
        q_probs = self.policy_net(states)
        actions = actions.view(batch_size, 1, 1)
        actions_mask = actions.expand(batch_size, 1, self.atoms)
        qa_probs = q_probs.gather(1, actions_mask).squeeze()

        with torch.no_grad():
            target_qa_probs = self.get_categorical(batch_size, next_states, rewards, is_terminal)

        qa_probs = qa_probs.clamp(min=1e-10)

        loss = - torch.sum(target_qa_probs * torch.log(qa_probs))
        return loss

    def get_categorical(self, batch_size, next_states, rewards, is_terminal):

        gamma = self.gamma

        # Compute probabilities p(x, a)
        probs = self.target_net(next_states)
        qs = torch.mul(probs, self.support.expand_as(probs))
        argmax_a = qs.sum(2).max(1)[1].unsqueeze(1).unsqueeze(1)
        action_mask = argmax_a.expand(batch_size, 1, self.atoms)
        qa_probs = probs.gather(1, action_mask).squeeze()

        # Mask gamma and reshape it torgether with rewards to fit p(x,a).
        rewards = rewards.expand_as(qa_probs)
        gamma = ((1 - is_terminal.float()) * gamma).expand_as(qa_probs)

        # Compute projection of the application of the Bellman operator.
        bellman_op = rewards + gamma * self.support.unsqueeze(0).expand_as(rewards)
        bellman_op = torch.clamp(bellman_op, self.v_min, self.v_max)

        # Compute categorical indices for distributing the probability

        m = self.m.fill_(0)
        b = (bellman_op - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        # Distribute probability
        """
        for i in range(batch_size):
            for j in range(self.atoms_no):
                uidx = u[i][j]
                lidx = l[i][j]
                m[i][lidx] = m[i][lidx] + qa_probs[i][j] * (uidx - b[i][j])
                m[i][uidx] = m[i][uidx] + qa_probs[i][j] * (b[i][j] - lidx)
        for i in range(batch_size):
            m[i].index_add_(0, l[i], qa_probs[i] * (u[i].float() - b[i]))
            m[i].index_add_(0, u[i], qa_probs[i] * (b[i] - l[i].float()))
        """
        # Optimized by https://github.com/tudor-berariu
        offset = torch.linspace(0, ((batch_size - 1) * self.atoms), batch_size, device=self.device)\
            .long()\
            .unsqueeze(1).expand(batch_size, self.atoms)
        m.view(-1).index_add_(0, (l + offset).view(-1),
                              (qa_probs * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1),
                              (qa_probs * (b - l.float())).view(-1))
        return m