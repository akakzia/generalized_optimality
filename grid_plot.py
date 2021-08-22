import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = '../experiments/'
directories = next(os.walk(path))[1]

data_rewards = np.empty((6, 6))
x = [0.74, 0.79, 0.84, 0.89, 0.94, 0.99]
k = 1
value_to_ids = {0.74: 0, 0.79: 1, 0.84: 2, 0.89: 3, 0.94: 4, 0.99: 4}
for directory in directories:
    gamma_1, gamma_2 = float(directory.split('_')[-2]), float(directory.split('_')[-1])
    print(gamma_1, gamma_2)
    directory_path = os.path.join(path, directory)
    data_run = pd.read_csv(directory_path + '/progress.csv')
    n = data_run.shape[0]
    last_reward = data_run['test SR'][n-3]
    stop = 1
    # last_reward = int(last_reward)
    data_rewards[value_to_ids[gamma_1], value_to_ids[gamma_2]] = last_reward
X, Y = np.meshgrid(x, x)
# fig = plt.figure()
# ax = fig.add_subplot(131)
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
cp = ax.pcolormesh(X, Y, data_rewards,  vmin=np.min(data_rewards), vmax=np.max(data_rewards), shading='nearest')
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Average Reward - HalfCheetah-v2')
ax.set_xlabel('Ɣ 1')
ax.set_ylabel('Ɣ 2')
ax.set_xticks(list(value_to_ids.keys()))
ax.set_yticks(list(value_to_ids.keys()))
plt.show()
stop = 1