import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical

import os
import numpy as np
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
policy_lr = 1e-2
value_lr = 1e-2
gamma = 0.99

# train
env = "CartPole"
policy = "REINFORCE_with_Baseline"
result_path = f"./results/{env}/{policy}"
model_path = f"./models/{env}/{policy}"
print_interval = 20
max_episode = int(5e2)


class Policy(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim, continuous=False):
		super(Policy, self).__init__()
		self.continuous = continuous

		self.fc1 = nn.Linear(state_dim, hidden_dim)
		if self.continuous:
			self.fc_mu = nn.Linear(hidden_dim, action_dim)
			self.fc_std = nn.Linear(hidden_dim, action_dim)
		else:
			self.fc2 = nn.Linear(hidden_dim, action_dim)

	
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
	def __init__(self, state_dim, hidden_dim):
		super(StateValue, self).__init__()
		
		self.fc1 = nn.Linear(state_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, 1)
		

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class REINFORCEWithBaseline():
	def __init__(self, state_dim, action_dim, max_action=None, policy_lr=1e-3, value_lr=1e-3, gamma=0.99, continuous=False):
		self.policy = Policy(state_dim ,action_dim, hidden_dim=32, continuous=continuous).to(device)
		self.value = StateValue(state_dim, hidden_dim=32).to(device)
		self.trajectory = []
		self.max_action = max_action
		self.gamma = gamma
		self.continuous = continuous

		self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
		self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)


	def select_action(self, state):
		state = torch.from_numpy(state).float().to(device)
		if self.continuous:
			mu, std = self.policy(state)
			dist = Normal(mu, std)
			action = dist.rsample()
			log_prob = dist.log_prob(action)
			return torch.tanh(action).detach().cpu().numpy() * self.max_action, log_prob
		
		else:
			prob = self.policy(state)
			m = Categorical(prob)
			action = m.sample()
			return action.detach().cpu().numpy(), m.log_prob(action)


	def add_data(self, data):
		self.trajectory.append(data)


	def train(self):
		G = 0
		self.policy_optimizer.zero_grad()
		self.value_optimizer.zero_grad()

		for r, log_prob, state in self.trajectory[::-1]:
			G = r + self.gamma * G
			value = self.value(torch.from_numpy(state).float().to(device))
			error = G - value
			# calculate state value network gradient
			value_loss = error * value
			value_loss.backward()
			# calculate policy network gradient
			policy_loss = -log_prob * G
			policy_loss.backward()
		
		# update network
		self.policy_optimizer.step()
		self.value_optimizer.step()
		self.trajectory = []


def pendulum():
	if not os.path.exists(result_path):
		os.makedirs(result_path)

	if os.path.exists(model_path):
		os.makedirs(model_path)

	# observation : (3,)
	# action : (1, ) 
	# action_bound : [-2, 2]
	env = gym.make("Pendulum-v1", render_mode="rgb_array")
	agent = REINFORCEWithBaseline(env.observation_space.shape[0], env.action_space.shape[0], max_action=2,\
						policy_lr=policy_lr, value_lr=value_lr, gamma=gamma, continuous=True)

	score = 0.
	episode_scores = np.empty((0,))

	for n_episode in range(1, max_episode+1):
		state, _ = env.reset()
		done = False
		truncated = False
	
		while not done and not truncated:
			action, log_prob = agent.select_action(state)
			next_state, reward, done, truncated, info = env.step(action)
			agent.add_data((reward, log_prob, state))
			state = next_state
			score += reward

		episode_scores = np.append(episode_scores, np.array([score]), axis=0)
		score = 0.
		agent.train()

		if n_episode % print_interval == 0:
			print("# of episode :{}, avg score : {}".format(n_episode, episode_scores[-print_interval:].mean()))
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
	# recorder = VideoRecorder(env=env, path=os.path.join(result_path, "video.mp4"))
	print(f"observation space : {env.observation_space}")
	print(f'action space: {env.action_space}')

	agent = REINFORCEWithBaseline(env.observation_space.shape[0], 2,\
						policy_lr=policy_lr, value_lr=value_lr, gamma=gamma, continuous=False)

	score = 0.
	episode_scores = np.empty((0,))

	for n_episode in range(1, max_episode+1):
		state, _ = env.reset()
		done = False
		truncated = False
	
		while not done and not truncated:
			action, prob = agent.select_action(state)
			next_state, reward, done, truncated, info = env.step(action)
			# recorder.capture_frame()
			agent.add_data((reward, prob, state))
			state = next_state
			score += reward

		episode_scores = np.append(episode_scores, np.array([score]), axis=0)
		score = 0.0
		agent.train()
		

		if n_episode % print_interval == 0:
			print("# of episode :{}, avg score : {}".format(n_episode, episode_scores[-print_interval:].mean()))
			np.save(f"{result_path}/scores", episode_scores)

	# recorder.close()
	env.close()

if __name__ == "__main__":
	cartpole()