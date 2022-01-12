class Solution:
    name = None
    
    def __init__(self, k, param_dict):
        self.k = k
        self.pred_values = self.init_pred_values()
        self.N = [0 for _ in range(self.k)]

    def init_pred_values(self):
        raise NotImplementedError()

    def choose_action(self):
        raise NotImplementedError()

    def update(self, action, reward):
        raise NotImplementedError()
