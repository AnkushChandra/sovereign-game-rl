import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DQN_HIDDEN_SIZES, DQN_LEARNING_RATE, DQN_GAMMA,
    DQN_BATCH_SIZE, DQN_BUFFER_SIZE, DQN_MIN_BUFFER,
    DQN_TARGET_UPDATE, DQN_GRAD_CLIP,
    DQN_EPS_START, DQN_EPS_END, DQN_EPS_DECAY_STEPS,
)


# ─────────────────────────────────────────────
# Q-Network (simple MLP)
# ─────────────────────────────────────────────

class QNetwork(nn.Module):

    def __init__(self, obs_dim, n_actions, hidden_sizes=DQN_HIDDEN_SIZES):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────

class ReplayBuffer:

    def __init__(self, capacity=DQN_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.as_tensor(np.array(states),      dtype=torch.float32, device=device)
        actions     = torch.as_tensor(actions,                dtype=torch.long,    device=device)
        rewards     = torch.as_tensor(rewards,                dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones       = torch.as_tensor(dones,                  dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────

class DQNAgent:

    def __init__(self, obs_dim, n_actions, device=None, seed=None):
        """Initialise networks, optimiser, and replay buffer."""
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        # Networks
        self.q_net     = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimiser
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=DQN_LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(DQN_BUFFER_SIZE)

        # Step counter (for ε schedule and target updates)
        self.total_steps = 0
        self.gradient_steps = 0

    # ─────────────────────────────────────
    # Exploration
    # ─────────────────────────────────────

    def epsilon(self):
        frac = min(1.0, self.total_steps / DQN_EPS_DECAY_STEPS)
        return DQN_EPS_START + frac * (DQN_EPS_END - DQN_EPS_START)

    def select_action(self, obs, greedy=False):
        self.total_steps += 1

        if not greedy and random.random() < self.epsilon():
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            q_values = self.q_net(obs_t.unsqueeze(0))
            return int(q_values.argmax(dim=1).item())

    # ─────────────────────────────────────
    # Memory + learning
    # ─────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, float(done))

    def learn(self):

        if len(self.buffer) < max(DQN_MIN_BUFFER, DQN_BATCH_SIZE):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            DQN_BATCH_SIZE, self.device,
        )

        # Current Q(s, a)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target:  r + γ · max_a' Q_target(s', a') · (1 − done)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + DQN_GAMMA * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), DQN_GRAD_CLIP)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % DQN_TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # ─────────────────────────────────────
    # Save / load
    # ─────────────────────────────────────

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state)
        self.target_net.load_state_dict(state)
