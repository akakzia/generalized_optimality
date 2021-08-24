import argparse
import gym
import envs
from sac_implementation.sac import SAC
from sac_implementation.generalized_sac import GSAC

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="DeceptiveSquare-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--gamma-1', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--gamma-2', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--episode-steps', type=int, default=200, metavar='N',
                    help='number of steps per episode (default: 100)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--update_frequency', type=int, default=10, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='save and test interval(default: 10)')

args = parser.parse_args()

env = gym.make(args.env_name)

# Agent
agent = GSAC(env.observation_space.shape[0], env.action_space, args)
# agent.load_model(actor_path='experiments/2021-08-23 19:38:12_SAC_0.99/models/actor',
#                  critic_path='experiments/2021-08-23 19:38:12_SAC_0.99/models/critic')
agent.load_model(actor_path='experiments/2021-08-23 20:13:25_GSAC_0.99_0.99/models/actor',
                 critic_1_path='experiments/2021-08-23 20:13:25_GSAC_0.99_0.99/models/critic_1',
                 critic_2_path='experiments/2021-08-23 20:13:25_GSAC_0.99_0.99/models/critic_2')
nb_demos = 20

avg_reward = 0
for episode in range(nb_demos):
    state = env.reset(evaluate=True)
    episode_reward = 0
    done = False
    for i in range(200):
        action = agent.select_action(state, evaluate=True)
        # action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        env.render()
        episode_reward += reward
        if done:
            break

        state = next_state
    avg_reward += episode_reward
avg_reward /= nb_demos

print('Average reward: {}'.format(avg_reward))