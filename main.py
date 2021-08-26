import datetime
import gym
import envs
import numpy as np
from mpi4py import MPI
import itertools
import torch
from sac_implementation.sac import SAC
from sac_implementation.generalized_sac import GSAC
from sac_implementation.replay_memory import ReplayMemory
from sac_implementation.arguments import get_args
import sac_implementation.logger as logger
from sac_implementation.utils import init_storage
from rollout import RolloutWorker

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

    # def rollout worker
    rollout_worker = RolloutWorker(env, agent, args)
    ################################################################################"
    # Training Loop
    # total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        max_reward_obtained = rollout_worker.play()
        if len(rollout_worker.memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = rollout_worker.policy.update_parameters(rollout_worker.memory, args.batch_size, updates)
                updates += 1
        if rollout_worker.total_steps > args.num_steps:
            break

        print("Episode: {}, total numsteps: {}, reward: {}".format(i_episode, rollout_worker.total_steps, round(max_reward_obtained, 2)))

        if i_episode % args.save_interval == 0 and args.eval is True:
            logger.info('\n\nElapsed steps #{}'.format(rollout_worker.total_steps))
            avg_reward = None
            for k in range(14):
                avg_reward = rollout_worker.eval(n=10, init=k)
                stats['test SR {}'.format(k)].append(np.around(avg_reward, 2))

            log_and_save(stats, i_episode, rollout_worker.total_steps, updates, avg_reward)

            if args.agent == 'GSAC':
                rollout_worker.policy.save_model(path=model_path)
            else:
                rollout_worker.policy.save_model(path=model_path)


    env.close()

def log_and_save(stats, i_episode, total_numsteps, updates, avg_reward):
    stats['episodes'].append(i_episode)
    stats['environment steps'].append(total_numsteps)
    stats['updates'].append(updates)
    for k, l in stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()

if __name__ == '__main__':
    # Get parameters
    args = get_args()

    launch(args)