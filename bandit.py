import numpy as np

class Bandit:
    def __init__(self, k):
        self.k = k
        self.values = self.init_values()
        self.optimal_actions = [action for action, value in enumerate(self.values) 
                                if value == max(self.values)]

    def init_values(self):
        return [np.random.normal(0, 1) for _ in range(self.k)]

    def reward(self, action):
        value = self.values[action]
        return np.random.normal(value, 1)

    def step(self, solution):
        action = solution.choose_action()
        reward = self.reward(action)
        solution.update(action, reward)
        return action, reward

    def is_optimal(self, action):
        return action in self.optimal_actions