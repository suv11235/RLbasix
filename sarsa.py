#!/usr/bin/env python3
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import random
from collections import deque

def evaluate_policy(env, Q, num_episodes=100):
    """
    Evaluate the performance of the learned policy
    """
    successes = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == 1.0:  # Successfully reached the goal
                successes += 1
            state = next_state
    return successes / num_episodes

def sarsa(env, num_episodes=5000, alpha=0.1, gamma=0.99,
          epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    """
    SARSA algorithm with improved learning parameters and monitoring
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_history = deque(maxlen=100)  # Track recent rewards
    success_rate_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        # Choose initial action using epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
            
        done = False
        while not done:
            # Take action and observe outcome
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Modify reward structure to encourage learning
            if reward == 0 and not done:
                reward = -0.01  # Small negative reward for each step
            elif reward == 0 and done:
                reward = -1  # Penalty for falling in a hole
            # Keep the original reward of 1 for reaching the goal
            
            # Choose next action using epsilon-greedy
            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA update rule
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            state, action = next_state, next_action
            episode_reward += reward
        
        # Store episode reward
        rewards_history.append(episode_reward)
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Evaluate policy every 100 episodes
        if (episode + 1) % 100 == 0:
            success_rate = evaluate_policy(env, Q)
            success_rate_history.append(success_rate)
            avg_reward = sum(rewards_history) / len(rewards_history)
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}, Epsilon: {epsilon:.2f}")
            
    return Q, success_rate_history

if __name__ == "__main__":
    # Create environment
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")
    
    # Train the agent
    Q, success_history = sarsa(env)
    
    print("\nLearned Q-table:")
    print(Q)
    
    print("\nFinal Success Rate:", success_history[-1])
    
    # Demonstrate learned policy
    print("\nDemonstrating learned policy:")
    state, _ = env.reset()
    env.render()
    done = False
    total_reward = 0
    
    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()
    
    print(f"Episode finished with reward: {total_reward}")
