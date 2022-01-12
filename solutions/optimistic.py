from .greedy import Greedy

class Optimistic(Greedy):
    name = "Optimistic greedy"

    def __init__(self, k, param_dict):
        self.init_value = param_dict["init_value"]
        if not param_dict.get("epsilon"):
            param_dict["epsilon"] = 0

        super().__init__(k, param_dict)

    def init_pred_values(self):
        return [self.init_value for _ in range(self.k)]