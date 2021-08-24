import datetime
import gym
import envs
import numpy as np
from mpi4py import MPI
import itertools
import torch
import pickle as pkl
from sac_implementation.sac import SAC
from sac_implementation.generalized_sac import GSAC
from sac_implementation.replay_memory import ReplayMemory
from sac_implementation.arguments import get_args
import sac_implementation.logger as logger
from sac_implementation.utils import init_storage

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

    # Set up logger
    logdir, model_path = init_storage(args)
    logger.configure(dir=logdir)
    logger.info(vars(args))

    # stats
    stats = dict()
    stats['episodes'] = []
    stats['environment steps'] = []
    stats['updates'] = []
    for k in range(14):
        stats['test SR {}'.format(k)] = []

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        state = env.reset(pos=np.random.randint(12))
        done = False

        # Perform one rollout
        while episode_steps < args.max_episode_steps:
            # if args.start_steps > total_numsteps:
            #     action = env.action_space.sample()  # Sample random action
            # else:
            action = agent.select_action(state)  # Sample action from policy

            next_state, reward, done, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward = max(reward, episode_reward)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            # mask = 1
            mask = 1 if episode_steps == args.max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

            # if done:
            #     state = env.reset()

        if len(memory) > args.start_steps:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                updates += 1
        if total_numsteps > args.num_steps:
            break

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % args.save_interval == 0 and args.eval is True:
            logger.info('\n\nElapsed steps #{}'.format(total_numsteps))
            episodes = 10
            for k in range(14):
                avg_reward = 0.
                for _ in range(episodes):
                    state = env.reset(pos=k)
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
                stats['test SR {}'.format(k)].append(np.around(avg_reward, 2))
            # state = env.reset()

            log_and_save(stats, i_episode, total_numsteps, updates, avg_reward)
            # print("----------------------------------------")
            # print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            # print("----------------------------------------")

            if args.agent == 'GSAC':
                agent.save_model(path=model_path)
            else:
                agent.save_model(path=model_path)


    env.close()
    stop = 1

def log_and_save(stats, i_episode, total_numsteps, updates, avg_reward):
    stats['episodes'].append(i_episode)
    stats['environment steps'].append(total_numsteps)
    stats['updates'].append(updates)
    # stats['test SR'].append(avg_reward)
    for k, l in stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()

if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)