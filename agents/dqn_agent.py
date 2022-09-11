import abc
import torch
import torch.nn.functional as F
from torch import TensorType
from networks import QNetwork
from agents.base_agent import BaseAgent

class DQNAgent(BaseAgent, abc.ABC):
	def __init__(self, input_channels, no_actions, gamma, batch_size):
		super().__init__(input_channels=input_channels, no_actions=no_actions, gamma=gamma)

		self.batch_size = batch_size
		self.policy_net = QNetwork(self.input_channels, self.no_actions)
		self.target_net = QNetwork(self.input_channels, self.no_actions)
		self.old_policy_net = QNetwork(self.input_channels, self.no_actions)

		self.policy_net.to(self.device)
		self.target_net.to(self.device)
		self.old_policy_net.to(self.device)
		self.old_policy_net.load_state_dict(self.policy_net.state_dict())


	def train(self, batch_size, states, next_states, actions, rewards, is_terminal):
		Q_s_a = self.policy_net(states).gather(1, actions)
		# Obtain max_{a} Q(S_{t+1}, a) of any non-terminal state S_{t+1}.  If S_{t+1} is terminal, Q(S_{t+1}, A_{t+1}) = 0.
		# Note: each row of the network's output corresponds to the actions of S_{t+1}.  max(1)[0] gives the max action
		# values in each row (since this a batch).  The detach() detaches the target net's tensor from computation graph so
		# to prevent the computation of its gradient automatically.  Q_s_prime_a_prime is of size (BATCH_SIZE, 1).

		# Get the indices of next_states that are not terminal
		none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=self.device)
		# Select the indices of each row
		none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)

		Q_s_prime_a_prime = torch.zeros(batch_size, 1, device=self.device)
		if len(none_terminal_next_states) != 0:
			Q_s_prime_a_prime[none_terminal_next_state_index] = self.target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)

		# Compute the target
		target = rewards + self.gamma * Q_s_prime_a_prime 
		loss = F.smooth_l1_loss(target, Q_s_a)
		return loss

	def evaluate(self, state: TensorType):
		with torch.no_grad():
			q_val, argmax_a = self.policy_net(state).max(1)
		
		return (q_val, argmax_a)

	def calculate_update_norm(self):
        
		total_norm = 0
		new_parameters = [p for p in self.policy_net.parameters() if p.grad is not None and p.requires_grad]
		old_paremeters = [p for p in self.old_policy_net.parameters()]
		for old, new in zip(old_paremeters, new_parameters):
			param_norm = (new.detach().data - old.detach().data).norm(2)
			total_norm += param_norm.item() ** 2
		total_norm = total_norm ** 0.5
		self.old_policy_net.load_state_dict(self.policy_net.state_dict())

		return total_norm