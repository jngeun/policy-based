import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import gym
import numpy as np
import cv2
import copy
import os
from pathlib import Path

from ..Agent.utils import conv2d_size_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

# Hyperparameters
batch_size = 256
discount =  0.99
tau = 1e-2
lr_pi = 1e-4
lr_q = 1e-3
lr_alpha = 1e-2
init_alpha = 1e-2
target_entropy = -1.0
update_num = 100

# train
env = "CarRacing"
policy = "SAC"
result_path = f"./results/{env}/{policy}"
model_path = f"./models/{env}/{policy}"
print_interval = 1
max_episode = int(5e2)


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e5)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim[2], state_dim[0], state_dim[1]))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim[2], state_dim[0], state_dim[1]))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class PolicyNet(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim, lr_pi, lr_alpha, init_alpha, target_entropy):
		super(PolicyNet, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=state_dim[-1], out_channels=16, kernel_size=5, stride=4)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
		self.dropout = nn.Dropout(0.2)

		convw = conv2d_size_out(state_dim[0], kernel_size=5, stride=4)
		convw = conv2d_size_out(convw, kernel_size=3, stride=2)

		self.linear_input_size = 3872

		self.fc1 = nn.Linear(self.linear_input_size, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, hidden_dim)

		self.fc_mu = nn.Linear(hidden_dim, action_dim)
		self.fc_std  = nn.Linear(hidden_dim, action_dim)
		self.optimizer = optim.Adam(self.parameters(), lr=lr_pi)

		self.log_alpha = torch.tensor(np.log(init_alpha))
		self.log_alpha.requires_grad = True
		self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
		self.target_entropy = -action_dim


	def forward(self, x):
		if len(x.shape) == 3:
			x = torch.unsqueeze(x, 0)
		# x = torch.transpose(x, 0, 2)
		x = F.relu(self.conv1(x))
		x = self.dropout(x)
		x = F.relu(self.conv2(x))
		x = self.dropout(x)
		x = x.reshape(-1, self.linear_input_size)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))

		mu = self.fc_mu(x)
		std = F.softplus(self.fc_std(x))
		dist = Normal(mu, std)
		action = dist.rsample()
		log_prob = dist.log_prob(action)
		real_action = torch.tanh(action)
		real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
		return real_action, real_log_prob

	
	def train_net(self, q1, q2, mini_batch):
		s, _, _, _, _ = mini_batch
		a, log_prob = self.forward(s)
		entropy = -self.log_alpha.exp() * log_prob

		q1_val, q2_val = q1(s,a), q2(s,a)
		q1_q2 = torch.cat([q1_val, q2_val], dim=1)
		min_q = torch.min(q1_q2, 1, keepdim=True)[0]

		loss = -min_q - entropy # for gradient ascent
		self.optimizer.zero_grad()
		loss.mean().backward()
		self.optimizer.step()

		self.log_alpha_optimizer.zero_grad()
		alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
		alpha_loss.backward()
		self.log_alpha_optimizer.step()


class QNet(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim, learning_rate, tau):
		super(QNet, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=state_dim[-1], out_channels=16, kernel_size=5, stride=4)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
		self.dropout = nn.Dropout(0.2)

		convw = conv2d_size_out(state_dim[0], kernel_size=5, stride=4)
		convw = conv2d_size_out(convw, kernel_size=3, stride=2)

		self.linear_input_size = 3872

		self.fc1 = nn.Linear(self.linear_input_size, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim+action_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, hidden_dim)
		self.fc4 = nn.Linear(hidden_dim, 1)

		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		self.tau = tau


	def forward(self, x, a):
		# x = torch.transpose(x, 0, 2)
		if len(x.shape) == 3:
			x = torch.unsqueeze(x, 0)
		x = F.relu(self.conv1(x))
		x = self.dropout(x)
		x = F.relu(self.conv2(x))
		x = self.dropout(x)
		x = x.reshape(-1, self.linear_input_size)
		
		embedding = F.relu(self.fc1(x))
		cat = torch.cat([embedding, a], dim=1)
		q = F.relu(self.fc2(cat))
		q = F.relu(self.fc3(q))
		q = self.fc4(q)
		return q


	def train_net(self, target, mini_batch):
		s, a, r, s_prime, done = mini_batch
		loss = F.smooth_l1_loss(self.forward(s, a) , target)
		self.optimizer.zero_grad()
		loss.mean().backward()
		self.optimizer.step()

	def soft_update(self, net_target):
		for param_target, param in zip(net_target.parameters(), self.parameters()):
			param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


class SAC():
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		batch_size=256,
		discount=0.98,
		tau=0.01,
		lr_pi=0.0005,
		lr_q=0.001,
		lr_alpha=0.001,
		init_alpha=0.01,
		target_entropy=-1.0,
		update_num=20):
			
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_action = max_action
		self.batch_size = batch_size
		self.discount = discount
		self.tau = tau
		self.lr_pi = lr_pi
		self.lr_q = lr_q
		self.lr_alpha = lr_alpha
		self.init_alpha = init_alpha
		self.target_entropy = target_entropy
		self.update_num = update_num

		self.q1 = QNet(state_dim, action_dim, 32, lr_q, tau).to(self.device)
		self.q2 = QNet(state_dim, action_dim, 32, lr_q, tau).to(self.device)
		self.q1_target = copy.deepcopy(self.q1)
		self.q2_target = copy.deepcopy(self.q2)
		self.pi = PolicyNet(state_dim, action_dim, 32, lr_pi, lr_alpha, init_alpha, target_entropy).to(self.device)

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)

	def select_action(self, state):
		state = torch.from_numpy(state).float().to(self.device)
		a, log_prob= self.pi(state)

		return a.detach().cpu().numpy() * self.max_action, log_prob


	def train(self, batch_size=256):
		for i in range(self.update_num):
				# Sample replay buffer
				mini_batch = self.replay_buffer.sample(batch_size)

				td_target = self.calc_target(self.pi, self.q1_target, self.q2_target, mini_batch)
				self.q1.train_net(td_target, mini_batch)
				self.q2.train_net(td_target, mini_batch)                               
				self.pi.train_net(self.q1, self.q2, mini_batch)
				self.q1.soft_update(self.q1_target)
				self.q2.soft_update(self.q2_target)


	def calc_target(self, pi, q1, q2, mini_batch):
		s, a, s_prime, r, done = mini_batch

		with torch.no_grad():
			a_prime, log_prob= self.pi(s_prime)
			entropy = (-self.pi.log_alpha.exp() * log_prob.sum(-1, keepdim=True))
			q1_val, q2_val = self.q1(s_prime,a_prime), self.q2(s_prime,a_prime)
			q1_q2 = torch.cat([q1_val, q2_val], dim=1)
			min_q = torch.min(q1_q2, 1, keepdim=True)[0]
			target = r + self.discount * done * (min_q + entropy)

		return target



class CarRacing():
	def __init__(self):
		# observation : (96, 96, 3)
		# action : (3, )
		self.env = gym.make("CarRacing-v2", domain_randomize=True, continuous=True, render_mode="rgb_array")
		self.figure_dir = Path("/home/user/policy-based/figure")
		self.observation_space = np.asarray([96, 96, 1])
		self.action_space = int(3)


	def step(self, action):
		next_state, reward, done, _, info = self.env.step(action)
		next_state = self.image_processing(next_state)
		return next_state, reward, done, _, info


	def reset(self):
		state, _ = self.env.reset()
		state = self.image_processing(state)
		return state, _


	def close(self):
		self.env.close()


	def image_processing(self, obs):
		obs = self.green_mask(obs)
		obs = self.gray_scale(obs)
		obs = self.blur_image(obs)
		cv2.imshow("obs", obs)
		cv2.waitKey(1)
		obs = self.canny_edge_detector(obs)
		# print(self.fine_middle_position(obs))

		return np.expand_dims(obs, 0)


	def green_mask(self, observation):
		#convert to hsv
		hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
		
		mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

		#slice the green
		imask_green = mask_green > 0
		green = np.zeros_like(observation, np.uint8)
		green[imask_green] = observation[imask_green]
		return(green)

	
	def gray_scale(self, observation):
		gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
		return gray


	def blur_image(self, observation):
		blur = cv2.GaussianBlur(observation, (5, 5), 0)
		return blur


	def canny_edge_detector(self, observation):
		canny = cv2.Canny(observation, 50, 150)
		return canny
	

	def fine_middle_position(self, observation):
		cropped = observation[63:65, 24:73]
		nz = cv2.findNonZero(cropped)
		try:
			middle = (nz[:,0,0].max() + nz[:,0,0].min())/2
		except:
			middle = 0

		return middle


def main():
	if not os.path.exists(result_path):
		os.makedirs(result_path)

	if os.path.exists(model_path):
		os.makedirs(model_path)
	# observation : (96, 96, 3)
	# action : (3, )
	env = CarRacing()

	kwargs = {
	"state_dim": env.observation_space,
	"action_dim": env.action_space,
	"max_action": 2,
	"batch_size": batch_size,
	"discount": discount,
	"lr_pi": lr_pi,
	"lr_q": lr_q,
	"lr_alpha": lr_alpha,
	"init_alpha": init_alpha,
	"target_entropy": target_entropy,
	"update_num": update_num
	}
	agent = SAC(**kwargs)

	score = 0.
	episode_scores = np.empty((0,))
	
	for n_episode in range(max_episode):
		state, _ = env.reset()
		done = False
		truncated = False
	
		while not done and not truncated:
			action, log_prob = agent.select_action(state)
			next_state, reward, done, truncated, info = env.step(action[0])
			agent.replay_buffer.add(state, action, next_state, reward, done)
			state = next_state
			score += reward

		episode_scores = np.append(episode_scores, np.array([score]), axis=0)
		score = 0.
		agent.train()

		if n_episode % print_interval == 0:
			print("# of episode :{}, avg score : {}".format(n_episode, episode_scores[-print_interval:].mean()))
			np.save(f"{result_path}/scores", episode_scores)

	env.close()

if __name__ == "__main__":
	main()