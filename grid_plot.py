import os
import gym
import envs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

env_name = 'DeceptiveSquare-v0'
env = gym.make(env_name)
directory_path = 'experiments/2021-08-31 15:14:40_SAC_0.99/'
positions = np.absolute(np.array(env._init_positions) / 4).astype(int)
if env_name == 'DeceptiveSquare-v0':
    data_sr = -100 * np.ones((5, 10))
    x = np.arange(10)
    y = np.arange(5)
else:
    data_sr = -np.ones((5, 6))
    x = np.arange(6)
    y = np.arange(5)
data_run = pd.read_csv(directory_path + 'progress.csv')
n = data_run.shape[0]
for i in range(1, len(positions)-1):
    data_sr[positions[i][0], positions[i][1]] = data_run['test SR {}'.format(i)][n-1]
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
fig.set_size_inches(5, 1.5)
cmap = plt.get_cmap('inferno')
cp = ax.pcolormesh(X, Y, data_sr, cmap=cmap, vmin=0, vmax=1, shading='nearest')
fig.colorbar(cp) # Add a colorbar to a plot
# ax.set_title('DeceptiveMaze-v0 Time steps={} x 10^4'.format(round(i*0.2, 2)))
plt.axis('off')
plt.show()
plt.close()
# plt.savefig('{}.jpg'.format(i))
# plt.close('all')
# stop = 1