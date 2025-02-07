#!/usr/bin/env python3
import gym
import numpy as np
import random

def q_learning(env, num_episodes=2000, alpha=0.8, gamma=0.95, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            best_next_action = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next_action - Q[state, action])
            state = next_state
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1} completed.")
    return Q

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=False)  # deterministic environment
    Q = q_learning(env)
    print("Learned Q-table:")
    print(Q)

