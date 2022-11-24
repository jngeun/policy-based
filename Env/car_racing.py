import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import gym
import numpy as np
from utils import conv2d_size_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Hyperparameters
lr = 0.0001
gamma = 0.99

print_interval = 20
max_episode = 500


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(Policy, self).__init__()
        output_channel = 64

        self.conv1 = nn.Conv2d(in_channels=state_dim[-1], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=output_channel, kernel_size=3, stride=1)

        convw = conv2d_size_out(state_dim[0], kernel_size=8, stride=4)
        convw = conv2d_size_out(convw, kernel_size=4, stride=2)
        convw = conv2d_size_out(convw, kernel_size=3, stride=1)

        linear_input_size = convw ** 2 * output_channel

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc_mu = nn.Linear(512, action_dim)
        self.fc_std = nn.Linear(512, action_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    
    def forward(self, x):
        x = torch.transpose(x, 0, 2)
        x = torch.unsqueeze(x, 0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))

        return mu, std


class REINFORCE():
    def __init__(self, state_dim, action_dim, lr, gamma):
        self.policy = Policy(state_dim ,action_dim, lr)
        self.data = []

    def select_action(self, state):
        mu, std = self.policy(state)
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        
        return torch.tanh(action), log_prob


    def save_data(self, data):
        self.data.append(data)

    def train(self):
        G = 0
        self.policy.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            G = r + gamma * G
            print(G)
            loss = -torch.log(prob) * G
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    # observation : (96, 96, 3)
    # action : (3, )
    env = gym.make("CarRacing-v2", domain_randomize=True, continuous=True, render_mode="rgb_array")
    agent = REINFORCE(env.observation_space.shape, env.action_space.shape[0],\
                        lr, gamma)

    score = 0.

    for n_episode in range(max_episode):
        state, _ = env.reset()
        done = False
    
        while not done:
            action, prob = agent.select_action(torch.from_numpy(state).float())
            next_state, reward, done, _, info = env.step(action.detach().cpu().numpy()[0])
            agent.save_data((reward, prob))
            state = next_state
            score += reward
            env.render()

        agent.train()

        if n_episode % print_interval == 0:
            print("# of episode :{}, avg score : {}".format(n_episode, score/print_interval))
            score = 0.0
    env.close()

if __name__ == "__main__":
    main()