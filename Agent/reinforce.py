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
lr = 1e-2
gamma = 0.99
batch_size = 256
taru = 0.01
actor_lr = 0.0005
critic_lr = 0.001
alpha_lr = 0.001
init_alpha = 0.01
target_entropy = -1.0
update_num = 100

# train
env = "CartPole"
policy = "REINFORCE"
result_path = f"./results/{env}/{policy}"
model_path = f"./models/{env}/{policy}"
print_interval = 20
max_episode = int(5e2)


class Policy(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim, continuous=False):
		super(Policy, self).__init__()
		self.continuous = continuous
		hidden_dim = int(hidden_dim)
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


class REINFORCE():
	def __init__(self, state_dim, action_dim, max_action=None, lr=1e-4, gamma=0.99, continuous=False):
		self.policy = Policy(state_dim ,action_dim, hidden_dim=32, continuous=continuous).to(device)
		self.trajectory = []
		self.max_action = max_action
		self.gamma = gamma
		self.continuous = continuous

		self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

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


	def add_data(self, trajectory):
		self.trajectory.append(trajectory)

	def train(self):
		G = 0
		self.optimizer.zero_grad()
		for r, log_prob in self.trajectory[::-1]:
			G = r + self.gamma * G
			loss = -log_prob * G
			loss.backward()
		self.optimizer.step()
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
	agent = REINFORCE(env.observation_space.shape[0], env.action_space.shape[0], max_action=2,\
						lr=lr, gamma=gamma, continuous=True)

	score = 0.
	episode_scores = np.empty((0,))

	for n_episode in range(1, max_episode+1):
		state, _ = env.reset()
		done = False
		truncated = False
	
		while not done and not truncated:
			action, log_prob = agent.select_action(state)
			next_state, reward, done, truncated, info = env.step(action)
			agent.add_data((reward, log_prob))
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

	agent = REINFORCE(env.observation_space.shape[0], 2,\
						lr=lr, gamma=gamma, continuous=False)

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
			agent.add_data((reward, prob))
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