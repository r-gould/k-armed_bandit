import random

from .solution import Solution

class Greedy(Solution):
    name = "Greedy"

    def __init__(self, k, param_dict):
        self.epsilon = param_dict["epsilon"]

        super().__init__(k, param_dict)

    def init_pred_values(self):
        return [0 for _ in range(self.k)]

    def choose_action(self):
        greedy_prob = random.random()
        if greedy_prob < self.epsilon:
            actions = [i for i in range(self.k)]
            return random.choice(actions)

        max_value = max(self.pred_values)
        greedy_actions = [action for action, value in enumerate(self.pred_values) 
                          if value == max_value]
        return random.choice(greedy_actions)

    def update(self, action, reward):
        value = self.pred_values[action]
        self.N[action] += 1
        N_a = self.N[action]
        self.pred_values[action] = value + (reward - value) / N_a