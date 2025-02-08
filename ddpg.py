#!/usr/bin/env python3
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import deque

# Patch NumPy to include bool8 if it's missing
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

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

def ddpg_training(env, num_episodes=500, batch_size=64, gamma=0.99, tau=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Adjusted hyperparameters: increased actor lr and more training episodes.
    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    target_actor = Actor(state_dim, action_dim, max_action).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    replay_buffer = deque(maxlen=100000)
    
    # To track performance
    all_rewards = []
    
    # Initial exploration noise scale; we'll decay this over episodes.
    noise_scale = 0.2

    def select_action(state, noise_scale):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = actor(state_tensor).cpu().data.numpy().flatten()
        action += noise_scale * np.random.randn(action_dim)
        return np.clip(action, -max_action, max_action)

    for episode in range(num_episodes):
        # Reset the environment (handles both Gym and Gymnasium APIs)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result  # Gymnasium API: (observation, info)
        else:
            state = reset_result       # Gym API

        episode_reward = 0
        done = False
        while not done:
            action = select_action(state, noise_scale)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)

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

        # Decay exploration noise
        noise_scale = max(noise_scale * 0.995, 0.05)
        all_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Noise Scale: {noise_scale:.3f}")

    # Plot training rewards
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Reward over Episodes')
    plt.grid(True)
    plt.show()

    return actor, critic

def evaluate(actor, env, num_episodes=5):
    """Visualize inference using the trained actor (no exploration noise)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ep in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result

        episode_reward = 0
        done = False
        while not done:
            env.render()
            time.sleep(0.02)  # Slow down for visualization
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = actor(state_tensor).cpu().data.numpy().flatten()
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            state = next_state
            episode_reward += reward

        print(f"Evaluation Episode {ep+1}, Reward: {episode_reward:.2f}")
    env.close()

if __name__ == "__main__":
    # Create a training environment (no rendering)
    train_env = gym.make('Pendulum-v1')
    actor, critic = ddpg_training(train_env, num_episodes=500)

    # Create a separate evaluation environment with rendering enabled.
    eval_env = gym.make('Pendulum-v1', render_mode='human')
    evaluate(actor, eval_env, num_episodes=5)
