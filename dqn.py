#!/usr/bin/env python3
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

# Monkey patch for numpy: define np.bool8 if it doesn't exist
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def dqn_training(env, num_episodes=500, batch_size=64, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    replay_buffer = deque(maxlen=10000)

    epsilon = epsilon_start

    def select_action(state):
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    for episode in range(num_episodes):
        # New gym API: env.reset() returns (observation, info)
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = select_action(state)
            # New gym API: env.step() returns 5 values
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                # Convert lists to numpy arrays before creating tensors
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                current_q = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + gamma * max_next_q * (1 - dones)
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    return policy_net

if __name__ == "__main__":
    # --- Training Phase ---
    # Create a training environment that does not open a window (render_mode="rgb_array")
    train_env = gym.make('CartPole-v1', render_mode="rgb_array")
    trained_policy = dqn_training(train_env)
    train_env.close()

    # --- Inference / Visualization Phase ---
    # Create a new environment for inference with a human-friendly render mode
    infer_env = gym.make('CartPole-v1', render_mode="human")
    print("\nStarting inference visualization. Close the window or stop the process to exit.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run a single inference episode
    state, _ = infer_env.reset()
    done = False
    total_reward = 0
    while not done:
        infer_env.render()  # This will show the visualization window

        # Select an action using the trained policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = trained_policy(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())

        # Step the environment using the new Gym API
        next_state, reward, terminated, truncated, _ = infer_env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

        # Slow down the visualization for clarity (optional)
        time.sleep(0.05)

    print(f"Inference episode completed. Total Reward: {total_reward:.2f}")
    infer_env.close()
