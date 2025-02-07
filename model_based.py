#!/usr/bin/env python3
import numpy as np
import random

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
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            Q[state] = {a: 0.0 for a in env.get_actions()}

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(env.get_actions())
            else:
                action = max(Q[state], key=Q[state].get)
            next_state, reward, done = env.step(action)
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
        print(f"Episode {episode+1} completed.")
    return Q

if __name__ == "__main__":
    env = GridWorld(size=5, start=(0,0), goal=(4,4))
    Q = dyna_q(env)
    # Display learned Q-values for each state
    for state in sorted(Q.keys()):
        print(f"State {state}: {Q[state]}")

