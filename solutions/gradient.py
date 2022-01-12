import random
import numpy as np

from math import exp
from .solution import Solution

class Gradient(Solution):
    name = "Gradient bandit"

    def __init__(self, k, param_dict):
        self.t = 1
        self.baseline = 0
        self.prefs = [0 for _ in range(k)]
        self.alpha = param_dict["alpha"]

        super().__init__(k, param_dict)

    def init_pred_values(self):
        return [None for _ in range(self.k)]

    def choose_action(self):
        probs = self.policy()
        actions = [i for i in range(self.k)]
        return np.random.choice(actions, p=probs)

    def update(self, action, reward):
        self.baseline = self.baseline + (reward - self.baseline) / self.t
        factor = self.alpha * (reward - self.baseline)
        probs = self.policy()

        for a in range(self.k):
            if a == action:
                self.prefs[a] = self.prefs[a] + factor * (1-probs[a])
                continue
            self.prefs[a] = self.prefs[a] - factor * probs[a]
        
        self.t += 1

    def policy(self):
        denominator = sum(map(exp, self.prefs))
        probs = [exp(pref)/denominator for pref in self.prefs]
        return probs