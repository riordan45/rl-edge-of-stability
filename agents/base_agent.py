import abc
import datetime
import torch
import torch.nn as nn
from torch import TensorType
from collections import namedtuple

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')

class BaseAgent(abc.ABC):

	def __init__(self, input_channels, no_actions, gamma):

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.input_channels = input_channels
		self.no_actions = no_actions
		self.gamma = gamma
		now = datetime.datetime.now()
		self.date_and_time = "{}-{}-{}_{}:{}:{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
	
	def update_target_network(self, network : nn.Module):
		"""
		Update the target net with the policy_net
		"""
		self.target_net.load_state_dict(network.state_dict())

	def get_minatar_state(self, s):
		return (torch.tensor(s, device=self.device).permute(2, 0, 1)).unsqueeze(0).float()
	
	def process_sample(self, sample):
		batch_sample = transition(*zip(*sample))

		states = torch.concat(batch_sample.state).to(self.device).float()
		next_states = torch.concat(batch_sample.next_state).to(self.device).float()
		actions = torch.concat(batch_sample.action).to(self.device).long()
		rewards = torch.concat(batch_sample.reward).to(self.device).float()
		is_terminal = torch.concat(batch_sample.is_terminal).to(self.device).float()

		return (states, next_states, actions, rewards, is_terminal)

	def calculate_gradient_norm(self, network : nn.Module):
		total_norm = 0
		parameters = [p for p in network.parameters() if p.grad is not None and p.requires_grad]
		for p in parameters:
			param_norm = p.grad.detach().data.norm(2)
			total_norm += param_norm.item() ** 2
		total_norm = total_norm ** 0.5
		
		return total_norm

	@abc.abstractmethod
	def train(self, batch_size : int, states : TensorType, next_states : TensorType, actions : TensorType, rewards : TensorType, is_terminal : TensorType
	) -> torch.Tensor :
		raise NotImplementedError

	@abc.abstractmethod
	def evaluate(self, state : TensorType):
		raise NotImplementedError
