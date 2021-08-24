import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = '../experiments/'
directories = next(os.walk(path))[1]

# data_rewards = -np.ones((19, 19))
data_rewards = -np.ones((11, 6))
# x = [0., 0.14, 0.19, 0.24, 0.29, 0.34, 0.39, 0.44, 0.49, 0.54, 0.59, 0.64, 0.69, 0.74, 0.79, 0.84, 0.89, 0.94, 0.99]
x = [0, 0.54, 0.59, 0.64, 0.69, 0.74, 0.79, 0.84, 0.89, 0.94, 0.99]
y = [0.74, 0.79, 0.84, 0.89, 0.94, 0.99]
k = 1
value_to_ids_x = {0.0: 0, 0.54: 1, 0.59: 2, 0.64: 3, 0.69: 4, 0.74: 5, 0.79: 6, 0.84: 7, 0.89: 8, 0.94: 9, 0.99: 10}
value_to_ids_y = {0.74: 0, 0.79: 1, 0.84: 2, 0.89: 3, 0.94: 4, 0.99: 5}
for i in range(50):
    for directory in directories:
        gamma_1, gamma_2 = float(directory.split('_')[-2]), float(directory.split('_')[-1])
        # print(gamma_1, gamma_2)
        directory_path = os.path.join(path, directory)
        data_run = pd.read_csv(directory_path + '/progress.csv')
        n = data_run.shape[0]
        last_reward = data_run['test SR'][i]
        # last_reward = int(last_reward)
        try:
            data_rewards[value_to_ids_x[gamma_1], value_to_ids_y[gamma_2]] = last_reward
            stop = 1
        except:
            pass
    X, Y = np.meshgrid(x, y)
    # fig = plt.figure()
    # ax = fig.add_subplot(131)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    cp = ax.pcolormesh(X, Y, np.transpose(data_rewards),  vmin=0, vmax=10, shading='nearest')
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('DeceptiveMaze-v0 Time steps={} x 10^4'.format(round(i*0.2, 2)))
    ax.set_xlabel('Ɣ 1')
    ax.set_ylabel('Ɣ 2')
    ax.set_xticks(list(value_to_ids_x.keys()))
    ax.set_yticks(list(value_to_ids_y.keys()))
    ax.set_xlim(0.515, 1.01)
    # plt.show()
    plt.savefig('{}.jpg'.format(i))
    plt.close('all')
    stop = 1