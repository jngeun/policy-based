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

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
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
	def __init__(self, state_dim, action_dim, hidden_dim, lr_pi, lr_alpha, init_alpha):
		super(PolicyNet, self).__init__()

		self.fc1 = nn.Linear(state_dim, hidden_dim)
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
		self.fc1 = nn.Linear(state_dim+action_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, 1)

		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		self.tau = tau


	def forward(self, x, a):
		q = torch.cat([x, a], dim=1)
		q = F.relu(self.fc1(q))
		q = F.relu(self.fc2(q))
		q = self.fc3(q)
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

		self.q1 = QNet(state_dim, action_dim, 128, lr_q, tau).to(self.device)
		self.q2 = QNet(state_dim, action_dim, 128, lr_q, tau).to(self.device)
		self.q1_target = copy.deepcopy(self.q1)
		self.q2_target = copy.deepcopy(self.q2)
		self.pi = PolicyNet(state_dim, action_dim, 128, lr_pi, lr_alpha, init_alpha).to(self.device)

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
		self.env = gym.make("CarRacing-v2", domain_randomize=False, continuous=True, render_mode="rgb_array")
		self.figure_dir = Path("/home/user/policy-based/figure")
		self.n_step = 0

		self.init_frames = 100
		self.agent_pos = (70, 50)
		self.crop_h = (60, self.agent_pos[0])
		self.crop_w = (30, 70)

		self.observation_space = int(2 * (self.crop_h[1]-self.crop_h[0])) # (left_lane, right_lane)
		self.action_space = int(1) # (linear, angular, brake)
		print(f' State_dim :{self.observation_space}, Action_dim :{self.action_space}')

		# Reward Gain
		self.middle_gain = 2
		self.speed_gain = 0.1
		self.brake_gain = 0.1
		

	def step(self, action):
		self.n_step += 1
		next_state, _, done, truncated, info = self.env.step(np.array([action, 0.01, 0], dtype=np.float32))
		next_state, truncated = self.image_processing(next_state)
		reward = self.get_reward(next_state, action, truncated)
		# Normalize
		next_state = (next_state - self.crop_w[0]) / (self.crop_w[1]-self.crop_w[0])
		return next_state, reward, done, truncated, info


	def reset(self):
		self.n_step= 0
		state, _ = self.env.reset()

		# pass init frames
		for _ in range(self.init_frames): 
			state, _, _, _, _ = self.env.step(np.zeros(3))

		state, truncated = self.image_processing(state)
		return state, _


	def close(self):
		self.env.close()


	def get_reward(self, state, action, truncated):
		if truncated:
			return -1
		
		# get high reward if agent is located in the middle of lane
		middle = state.mean()
		distance = 2 * abs(self.agent_pos[1] - middle) / (self.crop_w[1] - self.crop_w[0])
		distance_reward = self.middle_gain * (-distance + 0.25)
		# print(f'middle : {middle}, position : {self.agent_pos[1]}, distance : {distance}, reward : {distance_reward}')

		# The faster agent go, the higher the reward
		# speed_reward = self.speed_gain * action[1]
		# print(f'gas : {action[1]}, reward : {speed_reward}')

		# get penalty when brake is applied
		# brake_penalty = -self.brake_gain * action[2]

		print(f'distance_reward : {distance_reward}')
		return distance_reward

	def check_out_of_road(self, state):
		# check whether car is located on the green field
		is_green = state[self.agent_pos][1] > 150

		return is_green


	def image_processing(self, obs):
		mask = self.green_mask(obs)
		gray = self.gray_scale(mask)
		blur = self.blur_image(gray)
		edge = self.canny_edge_detector(blur)
		lane = self.find_lane(edge)

		cv2.imshow("mask", mask)
		cv2.imshow("gray", gray)
		cv2.imshow("blur", blur)
		cv2.imshow("edge", edge)
		cv2.waitKey(1)

		# If car run out of lane, terminate episode
		truncated = self.check_out_of_road(mask)

		self.show_env(obs, lane)
		return lane, truncated


	def show_env(self, obs, lane):
		# draw the range of image crop
		cv2.rectangle(obs, (self.crop_w[0], self.crop_h[0]), (self.crop_w[1], self.crop_h[1]), (249, 146, 69), thickness=3)
		
		
		for i, h in enumerate(range(self.crop_h[0], self.crop_h[1])):
			# draw lane point
			cv2.circle(obs, (lane[2*i], h), 3, (200, 30, 30))
			cv2.circle(obs, (lane[2*i+1], h), 3, (200, 30, 30))
			# draw middle point
			cv2.circle(obs, (int((lane[2*i] + lane[2*i+1]) / 2), h), 3, (30, 30, 200))

		obs = cv2.resize(obs,(600, 600))
		cv2.imshow("Car Racing", obs)
		cv2.waitKey(1)


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
	

	def find_lane(self, observation):
		lane_list = []

		for h in range(self.crop_h[0], self.crop_h[1]):
			cropped = observation[h, self.crop_w[0]:self.crop_w[1]]
			nz = cv2.findNonZero(cropped)
			if nz is None:
				(left, right) = self.crop_w
				lane_list.extend([left, right])
				continue
			left = nz[:,0,1].min() + self.crop_w[0]
			right = nz[:,0,1].max() + self.crop_w[0]
			if right - left < 5:
				if self.agent_pos[1] < left and self.agent_pos[1] < right:
					left = self.crop_w[0]
				else:
					right = self.crop_w[1]

			lane_list.extend([left, right])
		
		obs = np.array(lane_list, dtype=np.int32)
		
		return obs
	

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
	"max_action": 1,
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
			next_state, reward, done, truncated, info = env.step(action)
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