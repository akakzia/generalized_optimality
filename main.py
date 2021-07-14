import datetime
import gym
import envs
import numpy as np
from mpi4py import MPI
import itertools
import torch
from sac_implementation.sac import SAC
from sac_implementation.generalized_sac import GSAC
from torch.utils.tensorboard import SummaryWriter
from sac_implementation.replay_memory import ReplayMemory
from sac_implementation.arguments import get_args

def launch(args):
    rank = MPI.COMM_WORLD.Get_rank()

    # Environment
    env = gym.make(args.env_name)

    # Set random seeds for reproducibility
    env.seed(args.seed + rank)
    env.action_space.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Agent
    if args.agent == "SAC":
        agent = SAC(env.observation_space.shape[0], env.action_space, args)
    elif args.agent == "GSAC":
        agent = GSAC(env.observation_space.shape[0], env.action_space, args)
    else:
        raise NotImplementedError

    # Tensorboard
    if rank == 0:
        writer = SummaryWriter('runs/{}_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.agent,
                                                            args.env_name, args.policy,
                                                            "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        state = env.reset()
        done = False

        while not done and episode_steps < args.max_episode_steps:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy


            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == args.max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

            # if done:
            #     state = env.reset()

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

            if total_numsteps > args.num_steps:
                break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % args.save_interval == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                t = 0
                done = False
                while not done and t < args.max_episode_steps:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    t += 1
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes
            # state = env.reset()


            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
            agent.save_model(env_name=args.env_name, suffix='gen')

    env.close()

if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)