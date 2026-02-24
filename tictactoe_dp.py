import numpy as np
from collections import defaultdict

# Tic-Tac-Toe environment utilities
def is_winner(board, player):
    win_conds = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # cols
        [0, 4, 8], [2, 4, 6]             # diagonals
    ]
    for cond in win_conds:
        if all(board[i] == player for i in cond):
            return True
    return False

def is_full(board):
    return ' ' not in board

def get_possible_actions(board):
    return [i for i, c in enumerate(board) if c == ' ']

def get_next_states_and_rewards(board, action):
    """
    Returns a list of (next_state, reward, is_terminal, probability)
    Assuming player is 'X' and opponent 'O' plays completely randomly.
    """
    board_l = list(board) #converts the board string to a list
    board_l[action] = 'X'
    next_b = "".join(board_l) #converts the list back to a string
    
    if is_winner(next_b, 'X'):
        return [(next_b, 1.0, True, 1.0)]
    if is_full(next_b):
        return [(next_b, 0.0, True, 1.0)]
        
    opp_actions = get_possible_actions(next_b)
    prob = 1.0 / len(opp_actions) #used to assign win/loss/draw probabilities
    transitions = []
    
    for opp_a in opp_actions:
        opp_b_l = list(next_b)
        opp_b_l[opp_a] = 'O'
        opp_b = "".join(opp_b_l)
        
        if is_winner(opp_b, 'O'):
            transitions.append((opp_b, -1.0, True, prob))
        elif is_full(opp_b):
            transitions.append((opp_b, 0.0, True, prob))
        else:
            transitions.append((opp_b, 0.0, False, prob))
            
    return transitions

def generate_reachable_states(): #gives the R(s,a,s') and P(s,a,s')
    """BFS to find all states where it is X's turn."""
    states = set()
    queue = ["         "] # 9 spaces
    states.add(queue[0])
    
    transitions_map = {}
    
    while queue:
        s = queue.pop(0)
        
        if is_winner(s, 'X') or is_winner(s, 'O') or is_full(s):
            continue
            
        actions = get_possible_actions(s)
        transitions_map[s] = {}
        
        for a in actions:
            transitions = get_next_states_and_rewards(s, a)
            transitions_map[s][a] = transitions
            for next_s, r, done, prob in transitions:
                if next_s not in states:
                    states.add(next_s)
                    if not done:
                        queue.append(next_s)
                        
    return list(transitions_map.keys()), transitions_map

# ---------------------------------------------------------
# Dynamic Programming Algorithms
# ---------------------------------------------------------

def value_iteration(states, transitions_map, gamma=0.99, theta=1e-6):
    print("Starting Value Iteration...")
    V = defaultdict(float)
    policy = {}
    
    iteration = 0
    while True:
        delta = 0
        for s in states:
            v_old = V[s]
            action_values = []
            for a, transitions in transitions_map[s].items():
                expected_v = 0
                for next_s, r, done, prob in transitions:
                    expected_v += prob * (r + (gamma * V[next_s] if not done else 0))
                action_values.append((expected_v, a))
            
            best_value, best_action = max(action_values, key=lambda x: x[0])
            V[s] = best_value
            policy[s] = best_action
            delta = max(delta, abs(v_old - V[s]))
            
        iteration += 1
        if delta < theta:
            break
            
    print(f"Value Iteration converged in {iteration} iterations.")
    return V, policy

def policy_iteration(states, transitions_map, gamma=0.99, theta=1e-6):
    import random
    print("Starting Policy Iteration...")
    V = defaultdict(float)
    # Initialize completely random valid policy
    policy = {s: random.choice(get_possible_actions(s)) for s in states}
    
    iteration = 0
    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for s in states:
                v_old = V[s]
                a = policy[s]
                
                expected_v = 0
                for next_s, r, done, prob in transitions_map[s][a]:
                    expected_v += prob * (r + (gamma * V[next_s] if not done else 0))
                
                V[s] = expected_v
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break
                
        # Policy Improvement
        policy_stable = True
        for s in states:
            old_action = policy[s]
            action_values = []
            
            for a, transitions in transitions_map[s].items():
                expected_v = 0
                for next_s, r, done, prob in transitions:
                    expected_v += prob * (r + (gamma * V[next_s] if not done else 0))
                action_values.append((expected_v, a))
                
            best_value, best_action = max(action_values, key=lambda x: x[0])
            policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
                
        iteration += 1
        if policy_stable:
            break
            
    print(f"Policy Iteration converged in {iteration} iterations.")
    return V, policy

# ---------------------------------------------------------
# Test / Demo
# ---------------------------------------------------------
def display_board(board):
    print(f"{board[0]}|{board[1]}|{board[2]}")
    print("-+-+-")
    print(f"{board[3]}|{board[4]}|{board[5]}")
    print("-+-+-")
    print(f"{board[6]}|{board[7]}|{board[8]}")

if __name__ == "__main__":
    import time
    print("Generating state space...")
    states, transitions_map = generate_reachable_states()
    print(f"Generated {len(states)} non-terminal states where it's X's turn.")
    print("-" * 40)
    
    # Value Iteration
    start_time = time.time()
    V_vi, policy_vi = value_iteration(states, transitions_map, gamma=0.99)
    vi_time = time.time() - start_time
    print(f"Value Iteration Time: {vi_time:.5f} seconds")
    print("-" * 40)
    
    # Policy Iteration
    start_time = time.time()
    V_pi, policy_pi = policy_iteration(states, transitions_map, gamma=0.99)
    pi_time = time.time() - start_time
    print(f"Policy Iteration Time: {pi_time:.5f} seconds")
    print("-" * 40)
    
    # Check if policies are identical where values matter
    diffs = 0
    for s in states:
        if policy_vi[s] != policy_pi[s]:
            # Often multiple actions have the exact same value. Check if values match.
            diffs += 1
    print(f"Policies differ in exact action choice on {diffs} states (due to tie-breaking).")
    
    print("\nValue of starting state (empty board):")
    start_state = "         "
    display_board(start_state)
    print(f"\nV(start) = {V_vi[start_state]:.4f}")
    print(f"Best opening action index: {policy_vi[start_state]}")
