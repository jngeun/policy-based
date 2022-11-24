import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical

import os
import numpy as np
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
lr = 0.0002
gamma = 0.98

# train
env = "Pendulum"
policy = "REINFORCE_with_baseline"
result_path = f"./results/{env}/{policy}"
model_path = f"./models/{env}/{policy}"
print_interval = 20
max_episode = int(1e5)


class Policy(nn.Module):
	def __init__(self, state_dim, action_dim, lr, continuous=False):
		super(Policy, self).__init__()
		self.continuous = continuous

		self.fc1 = nn.Linear(state_dim, 512)
		if self.continuous:
			self.fc_mu = nn.Linear(512, action_dim)
			self.fc_std = nn.Linear(512, action_dim)
		else:
			self.fc2 = nn.Linear(512, action_dim)
		
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

	
	def forward(self, x):
		if self.continuous:
			x = F.relu(self.fc1(x))
			mu = self.fc_mu(x)
			std = F.softplus(self.fc_std(x))
			return mu, std
		else:
			x = F.relu(self.fc1(x))
			x = F.softmax(self.fc2(x), dim=0)
			return x


class StateValue(nn.Module):
	def __init__(self, state_dim, lr):
		super(StateValue, self).__init__()

		self.fc1 = nn.Linear(state_dim, 512)
		self.fc2 = nn.Linear(512, 1)
		
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class ReinforceWithBaseline():
	def __init__(self, state_dim, action_dim, max_action=None, lr=0.0001, gamma=0.99, continuous=False):
		self.policy = Policy(state_dim ,action_dim, lr, continuous).to(device)
		self.data = []
		self.continuous = continuous
		self.max_action = max_action

	def select_action(self, state):
		if self.continuous:
			mu, std = self.policy(state)
			dist = Normal(mu, std)
			action = dist.rsample()
			log_prob = dist.log_prob(action)
			return torch.tanh(action)* self.max_action, log_prob
		
		else:
			prob = self.policy(state)
			m = Categorical(prob)
			a = m.sample()
			return a, prob[a]


	def save_data(self, data):
		self.data.append(data)

	def train(self):
		G = 0
		self.policy.optimizer.zero_grad()
		for r, log_prob in self.data[::-1]:
			G = r + gamma * G
			loss = -log_prob * G
			loss.backward()
		self.policy.optimizer.step()
		self.data = []


def pendulum():
	if not os.path.exists(result_path):
		os.makedirs(result_path)

	if os.path.exists(model_path):
		os.makedirs(model_path)

	# observation : (3,)
	# action : (1, ) 
	# action_bound : [-2, 2]
	env = gym.make("Pendulum-v1", render_mode="rgb_array")
	agent = ReinforceWithBaseline(env.observation_space.shape[0], env.action_space.shape[0], max_action=2,\
						lr=lr, gamma=gamma, continuous=True)

	score = 0.
	episode_scores = []

	for n_episode in range(max_episode):
		state, _ = env.reset()
		done = False
		truncated = False
	
		while not done and not truncated:
			action, prob = agent.select_action(torch.from_numpy(state).float().to(device))
			next_state, reward, done, truncated, info = env.step(action.detach().cpu().numpy())
			agent.save_data((reward, prob))
			state = next_state
			score += reward

		episode_scores.append(score)
		score = 0.
		agent.train()

		if n_episode % print_interval == 0:
			print("# of episode :{}, avg score : {}".format(n_episode, episode_scores[-1]))
			np.save(f"{result_path}/scores", episode_scores)

	env.close()


def cartpole():

	if not os.path.exists(result_path):
		os.makedirs(result_path)

	if os.path.exists(model_path):
		os.makedirs(model_path)

	# observation : (4,)
	# action : discrete(2)
	env = gym.make("CartPole-v1", render_mode="rgb_array")
	print(f"observation space : {env.observation_space}")
	print(f'action space: {env.action_space}')

	agent = REINFORCE(env.observation_space.shape[0], 2,\
						lr=lr, gamma=gamma, continuous=False)

	score = 0.
	episode_scores = []

	for n_episode in range(max_episode):
		state, _ = env.reset()
		done = False
		truncated = False
	
		while not done and not truncated:
			action, prob = agent.select_action(torch.from_numpy(state).float().to(device))
			next_state, reward, done, truncated, info = env.step(action.detach().cpu().numpy())
			agent.save_data((reward, prob))
			state = next_state
			score += reward

		episode_scores.append(score)
		score = 0.0
		agent.train()
		

		if n_episode % print_interval == 0:
			print("# of episode :{}, avg score : {}".format(n_episode, episode_scores[-1]))
			np.save(f"{result_path}/scores", episode_scores)
			
	env.close()

if __name__ == "__main__":
	pendulum()