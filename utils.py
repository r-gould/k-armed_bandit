import matplotlib.pyplot as plt

def plot(solutions, rewards, accs):
    sol_classes = list(zip(*solutions))[0]
    sol_strs = [sol_class.name for sol_class in sol_classes]
    epoch_axis = [i for i in range(1, len(rewards[0])+1)]
    colors = ["red", "green", "blue", "orange"]

    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    for i, reward_data in enumerate(rewards):
        plt.plot(epoch_axis, reward_data, color=colors[i], label=sol_strs[i])
    plt.legend(loc="upper left")
    plt.show()

    plt.xlabel("Epochs")
    plt.ylabel("Optimal action %")
    for i, acc_data in enumerate(accs):
        plt.plot(epoch_axis, acc_data, color=colors[i], label=sol_strs[i])
    plt.legend(loc="upper left")
    plt.show()