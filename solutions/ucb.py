import random

from .solution import Solution

class UCB(Solution):
    name = "UCB"

    def __init__(self, k, param_dict):
        self.t = 1
        self.confidence = param_dict["confidence"]

        super().__init__(k, param_dict)

    def init_pred_values(self):
        return [0 for _ in range(self.k)]

    def choose_action(self):
        action_bounds = [None for _ in range(self.k)]

        for action in range(self.k):
            value = self.pred_values[action]
            N_a = self.N[action]
            variance = float("inf")
            if N_a > 0:
                variance = self.confidence * (self.t / N_a) ** 0.5
            
            action_bounds[action] = value + variance

        max_bound = max(action_bounds)
        confident_actions = [action for action, value in enumerate(action_bounds) 
                             if value == max_bound]
        return random.choice(confident_actions)

    def update(self, action, reward):
        value = self.pred_values[action]
        self.N[action] += 1
        N_a = self.N[action]
        self.pred_values[action] = value + (reward - value) / N_a