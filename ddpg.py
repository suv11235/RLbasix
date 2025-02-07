#!/usr/bin/env python3
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Actor network: maps state to action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action * self.max_action

# Critic network: maps state-action pair to Q-value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def ddpg_training(env, num_episodes=200, batch_size=64, gamma=0.99, tau=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, action_dim, max_action).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    replay_buffer = deque(maxlen=100000)

    def select_action(state, noise_scale=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor(state_tensor).cpu().data.numpy().flatten()
        action += noise_scale * np.random.randn(action_dim)  # exploration noise
        return np.clip(action, -max_action, max_action)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                # Critic update
                with torch.no_grad():
                    next_actions = target_actor(next_states)
                    target_q = target_critic(next_states, next_actions)
                    target_value = rewards + gamma * target_q * (1 - dones)
                current_q = critic(states, actions)
                critic_loss = nn.MSELoss()(current_q, target_value)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor update
                actor_loss = -critic(states, actor(states)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Soft-update target networks
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")

    return actor, critic

if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    actor, critic = ddpg_training(env)

