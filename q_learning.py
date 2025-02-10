#!/usr/bin/env python3
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import random
from collections import deque
import json
import time  # For adding delays during visualization

def q_learning(env, num_episodes=5000, alpha=0.1, gamma=0.99,
               epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    """
    Modified Q-learning with better hyperparameters and reward shaping
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_history = deque(maxlen=100)  # Track last 100 episodes
    all_rewards = []  # Track all episode rewards for visualization
    
    # Initialize success rate tracking
    success_rate = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Reward shaping: penalize ending without success and slight penalty for each step
            shaped_reward = reward
            if done and not reward:
                shaped_reward = -1
            elif not done:
                shaped_reward = -0.01
            
            best_next_value = np.max(Q[next_state])
            Q[state, action] += alpha * (shaped_reward + gamma * best_next_value - Q[state, action])
            
            state = next_state
            episode_reward += reward
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(episode_reward)
        all_rewards.append(episode_reward)  # Store reward for visualization
        success_rate = sum(rewards_history) / len(rewards_history)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Success rate: {success_rate:.2f}, Epsilon: {epsilon:.3f}")
    
    return Q, all_rewards

def evaluate_policy(env, Q, num_episodes=100):
    """
    Evaluate the learned policy without exploration
    """
    success_count = 0
    paths = []  # Store paths for visualization
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        path = [state]  # Track states visited in this episode
        
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            path.append(state)
            
            if reward == 1.0:
                success_count += 1
                break
        
        paths.append(path)
    
    success_rate = success_count / num_episodes
    return success_rate, paths

def get_optimal_path(Q):
    """Get the optimal path according to the Q-table"""
    state = 0  # Start state
    path = [state]
    
    while state != 15 and len(path) < 20:  # Prevent infinite loops
        action = np.argmax(Q[state])
        if action == 0:  # LEFT
            next_state = state if state % 4 == 0 else state - 1
        elif action == 1:  # DOWN
            next_state = state + 4 if state + 4 < 16 else state
        elif action == 2:  # RIGHT
            next_state = state if state % 4 == 3 else state + 1
        else:  # UP
            next_state = state - 4 if state - 4 >= 0 else state
            
        state = next_state
        path.append(state)
        
        if state == 15:  # Goal reached
            break
            
    return path

def visualize_inference(env, Q, delay=1.0):
    """
    Run one episode using the learned policy and render each step.
    The 'delay' parameter controls the pause (in seconds) between steps.
    """
    state, _ = env.reset()
    done = False
    print("Starting inference visualization...\n")
    
    # Initial render
    rendered_output = env.render()
    if rendered_output is not None:
        print(rendered_output)
    time.sleep(delay)
    
    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Render the current state; depending on the render_mode, this may print text or show a window.
        rendered_output = env.render()
        if rendered_output is not None:
            print(rendered_output)
        time.sleep(delay)
    
    # Final render for the terminal state
    env.render()
    print("Inference visualization complete.")

if __name__ == "__main__":
    # Train the agent using a headless environment
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    Q, rewards_history = q_learning(env)
    
    # Evaluate the learned policy
    success_rate, eval_paths = evaluate_policy(env, Q)
    print(f"\nFinal evaluation success rate: {success_rate:.2%}")
    
    # Get optimal path
    optimal_path = get_optimal_path(Q)
    
    # Create visualization data
    viz_data = {
        'rewards': rewards_history,
        'success_rate': float(success_rate),
        'q_table': Q.tolist(),
        'optimal_path': optimal_path
    }
    
    # Save visualization data to a JSON file
    with open('q_learning_results.json', 'w') as f:
        json.dump(viz_data, f)
    
    # For inference visualization, create a new environment with a render mode.
    # 'human' mode typically opens a window (if supported), otherwise consider using 'ansi'
    viz_env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")
    visualize_inference(viz_env, Q, delay=1.0)
    
    # Close the visualization environment properly
    viz_env.close()
