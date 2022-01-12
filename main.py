from bandit import Bandit
from solutions.greedy import Greedy
from solutions.optimistic import Optimistic
from solutions.ucb import UCB
from solutions.gradient import Gradient
from utils import plot

def main(k, runs, epochs, solutions):
    rewards = [[0 for _ in range(epochs)] for _ in range(len(solutions))]
    accs = [[0 for _ in range(epochs)] for _ in range(len(solutions))]

    for sol_num, (sol_class, param_dict) in enumerate(solutions):
        for run in range(runs):
            bandit = Bandit(k)
            solution = sol_class(k, param_dict)
            for epoch in range(epochs):
                action, reward = bandit.step(solution)
                is_optimal = bandit.is_optimal(action)

                rewards[sol_num][epoch] += reward / runs
                accs[sol_num][epoch] += is_optimal / runs
    
    plot(solutions, rewards, accs)

if __name__ == "__main__":
    solutions = [
        [Greedy, {"epsilon" : 0.1}],
        [Optimistic, {"init_value" : 5}],
        [UCB, {"confidence" : 2}],
        [Gradient, {"alpha" : 0.1}],
    ]
    k = 10
    runs = 2000
    epochs = 1000
    
    main(k, runs, epochs, solutions)