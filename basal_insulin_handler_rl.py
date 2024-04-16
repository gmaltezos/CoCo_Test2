import numpy as np

total_reward = 0.0 # Total reward
gamma = 0.95 # Discount Factor
# Simulation
n_episodes = 200
max_length_episode = 100

import numpy as np

class TabularQLearning:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_space_size, action_space_size))

    # Epsilon-greedy policy
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.q_table.shape[1])  # Explore: choose random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: choose action with highest Q-value

    def update_q_table(self, state, action, reward, next_state):
        # TD Learning
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

# Example usage
# Define state and action space sizes
state_space_size = 10
action_space_size = 4

# Initialize Q-learning agent
agent = TabularQLearning(state_space_size, action_space_size)

# Simulate the system
num_episodes = 1000
for episode in range(num_episodes):
    state = xsmiple
    done = False
    while not done:
        action = agent.choose_action(state)  # Choose action
        next_state, reward, done, _ = simulate  # Take action
        agent.update_q_table(state, action, reward, next_state)  # Update Q-table
        state = next_state 

