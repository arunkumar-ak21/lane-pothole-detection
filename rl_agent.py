import numpy as np
import random

class RLAgent:
    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.2):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon 
        self.action_size = 3  # 0: no alert, 1: soft alert, 2: strong alert
        self.q_table = {}

    def _key(self, state):
        return tuple([round(float(x), 2) for x in state])

    def choose(self, state):
        key = self._key(state)
        if random.random() < self.epsilon or key not in self.q_table:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[key]))

    def learn(self, state, action, reward, next_state):
        key = self._key(state)
        next_key = self._key(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        old_val = self.q_table[key][action]
        next_max = np.max(self.q_table[next_key])

        self.q_table[key][action] = old_val + self.lr * (
            reward + self.gamma * next_max - old_val
        )
