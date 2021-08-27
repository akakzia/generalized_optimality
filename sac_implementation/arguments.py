import argparse
import numpy as np
from mpi4py import MPI


"""
Training hyper-parameters
"""

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="DeceptiveSquare-v0",
                        help='Mujoco Gym environment (default: DeceptiveSquare-v0)')
    parser.add_argument('--seed', type=int, default=np.random.randint(1e5), metavar='N',
                        help='random seed (default: 123456)')

    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--agent', default="GSAC",
                        help='The RL algorithm to be used for training(default: SAC)')

    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')

    parser.add_argument('--gamma-1', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--gamma-2', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--depth', type=int, default=2, metavar='G',
                        help='Number of discount factors to be used')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--max-episode-steps', type=int, default=100, metavar='N',
                        help='number of steps per episode (default: 100)')
    parser.add_argument('--num_steps', type=int, default=200001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')

    parser.add_argument('--update-frequency', type=int, default=1, metavar='N',
                        help='Frequency update between gamma 1 and 2')

    parser.add_argument('--updates_per_step', type=int, default=100, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=-1, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=2, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')

    parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                        help='save and test interval(default: 10)')

    parser.add_argument('--save-dir', default="experiments",
                        help='')

    parser.add_argument('--init-zero', type=bool, default=False,
                        help='If true, initializes first critic to zero')

    args = parser.parse_args()

    return args