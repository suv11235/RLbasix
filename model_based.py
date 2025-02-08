#!/usr/bin/env python3
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
import time

class GridWorld:
    def __init__(self, size=5, start=(0, 0), goal=(4, 4)):
        self.size = size
        self.start = start
        self.goal = goal
        self.reset()
        
    def reset(self):
        self.agent_pos = list(self.start)
        return tuple(self.agent_pos)
        
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        x, y = self.agent_pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < self.size - 1:
            y += 1
        elif action == 2 and x < self.size - 1:
            x += 1
        elif action == 3 and y > 0:
            y -= 1
        self.agent_pos = [x, y]
        reward = 10 if (x, y) == self.goal else -1
        done = (x, y) == self.goal
        return (x, y), reward, done
        
    def get_actions(self):
        return [0, 1, 2, 3]

def dyna_q(env, num_episodes=50, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=5):
    # Initialize Q-table and model (both are dictionaries)
    Q = {}
    model = {}
    episode_rewards = []  # Track rewards for visualization
    
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            Q[state] = {a: 0.0 for a in env.get_actions()}
            
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(env.get_actions())
            else:
                action = max(Q[state], key=Q[state].get)
                
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Real experience update (similar to Q-learning)
            best_next = max(Q[next_state].values())
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
            
            # Update model with the observed transition
            model[(state, action)] = (next_state, reward)
            
            # Planning: simulate updates from the model
            for _ in range(planning_steps):
                s_a = random.choice(list(model.keys()))
                s_model, a_model = s_a
                next_s_model, r_model = model[s_a]
                best_next_model = max(Q[next_s_model].values())
                Q[s_model][a_model] += alpha * (r_model + gamma * best_next_model - Q[s_model][a_model])
                
            state = next_state
            
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1} completed. Total reward: {total_reward}")
        
    return Q, episode_rewards

def visualize_policy(env, Q):
    """Visualize the learned policy with arrows showing the best action at each state"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    for i in range(env.size + 1):
        ax.plot([0, env.size], [i, i], 'k-')
        ax.plot([i, i], [0, env.size], 'k-')
        
    # Mark start and goal
    ax.add_patch(Rectangle((env.start[1], env.start[0]), 1, 1, facecolor='green', alpha=0.3))
    ax.add_patch(Rectangle((env.goal[1], env.goal[0]), 1, 1, facecolor='red', alpha=0.3))
    
    # Draw arrows for policy
    action_to_direction = {
        0: (0, 0.4),    # up
        1: (0.4, 0),    # right
        2: (0, -0.4),   # down
        3: (-0.4, 0)    # left
    }
    
    for state in Q:
        x, y = state
        best_action = max(Q[state], key=Q[state].get)
        dx, dy = action_to_direction[best_action]
        ax.arrow(y + 0.5, x + 0.5, dy, dx,
                head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    ax.set_title("Learned Policy (arrows show best action)")
    ax.grid(True)
    plt.show()

def visualize_episode(env, Q):
    """Simulate and visualize one episode using the learned policy"""
    state = env.reset()
    path = [state]
    total_reward = 0
    done = False
    
    while not done:
        action = max(Q[state], key=Q[state].get)
        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(next_state)
        state = next_state
    
    # Plot the path
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    for i in range(env.size + 1):
        ax.plot([0, env.size], [i, i], 'k-')
        ax.plot([i, i], [0, env.size], 'k-')
    
    # Mark start and goal
    ax.add_patch(Rectangle((env.start[1], env.start[0]), 1, 1, facecolor='green', alpha=0.3))
    ax.add_patch(Rectangle((env.goal[1], env.goal[0]), 1, 1, facecolor='red', alpha=0.3))
    
    # Plot path
    path = np.array(path)
    ax.plot(path[:, 1] + 0.5, path[:, 0] + 0.5, 'b-', linewidth=2, label='Path')
    ax.plot(path[:, 1] + 0.5, path[:, 0] + 0.5, 'bo')
    
    ax.set_title(f"Agent Path (Total Reward: {total_reward})")
    ax.grid(True)
    ax.legend()
    plt.show()

def plot_learning_curve(rewards):
    """Plot the learning curve showing episode rewards over time"""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Create environment and train
    env = GridWorld(size=5, start=(0,0), goal=(4,4))
    Q, episode_rewards = dyna_q(env, num_episodes=100)  # Increased episodes for better learning
    
    # Display learned Q-values for each state
    print("\nLearned Q-values:")
    for state in sorted(Q.keys()):
        print(f"State {state}: {Q[state]}")
    
    # Visualize the results
    print("\nGenerating visualizations...")
    
    # Plot learning curve
    plot_learning_curve(episode_rewards)
    
    # Visualize policy
    visualize_policy(env, Q)
    
    # Simulate and visualize an episode
    visualize_episode(env, Q)
