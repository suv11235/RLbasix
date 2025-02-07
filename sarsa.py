#!/usr/bin/env python3
import gym
import numpy as np
import random

def sarsa(env, num_episodes=2000, alpha=0.8, gamma=0.95, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(num_episodes):
        state = env.reset()
        # Choose initial action using epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1} completed.")
    return Q

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=False)
    Q = sarsa(env)
    print("Learned Q-table from SARSA:")
    print(Q)

