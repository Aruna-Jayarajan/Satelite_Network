# === model2.py ===

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from collections import deque

# === RL Head ===
class SatelliteRLHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(SatelliteRLHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, satellite_embeddings):
        q_values = self.mlp(satellite_embeddings)
        return q_values

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(torch.stack, zip(*batch))
        return states, actions, rewards, next_states
    
    def __len__(self):
        return len(self.buffer)

# === DQN Agent ===
class DQNAgent:
    def __init__(self, gnn_model, rl_head, action_dim, device):
        self.device = device
        self.gnn_model = gnn_model.to(device)
        self.rl_head = rl_head.to(device)
        self.target_rl_head = copy.deepcopy(rl_head).to(device)
        self.action_dim = action_dim
        
        self.optimizer = optim.Adam(
            list(self.gnn_model.parameters()) + list(self.rl_head.parameters()), 
            lr=0.0005
        )
        
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1  # start exploration rate

    def select_action(self, sat_embeddings):
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (sat_embeddings.shape[0],)).to(self.device)
        else:
            q_values = self.rl_head(sat_embeddings)
            return q_values.argmax(dim=1)

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        
        # Current Q
        q_values = self.rl_head(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q
        with torch.no_grad():
            next_q_values = self.target_rl_head(next_states)
            max_next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + self.gamma * max_next_q_value
        
        # TD Loss
        loss = nn.MSELoss()(q_value, expected_q_value)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network slowly
        for target_param, param in zip(self.target_rl_head.parameters(), self.rl_head.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
