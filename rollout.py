import numpy as np
from sac_implementation.replay_memory import ReplayMemory

class RolloutWorker:
    def __init__(self, env, policy, args):
        self.env = env
        self.policy = policy
        self.args = args

        # Memory
        self.memory = ReplayMemory(args.replay_size, args.seed)

        self.total_steps = 0


    def play(self):
        episode_reward = 0
        episode_steps = 0
        state = self.env.reset(pos=np.random.randint(14))

        # Perform one rollout
        while episode_steps < self.args.max_episode_steps:
            if self.args.start_steps > self.total_steps:
                action = self.env.action_space.sample()  # Sample random action
            else:
                action = self.policy.select_action(state)  # Sample action from policy

            next_state, reward, done, _ = self.env.step(action)  # Step
            episode_steps += 1
            self.total_steps += 1
            episode_reward = max(reward, episode_reward)

            mask = 1 if episode_steps == self.args.max_episode_steps else float(not done)

            self.memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

            if done:
                state = self.env.reset(pos=np.random.randint(14))

        return episode_reward

    def eval(self, n=1, init=0):
        avg_reward = 0.
        for _ in range(n):
            state = self.env.reset(pos=init)
            episode_reward = 0
            t = 0
            done = False
            while not done and t < self.args.max_episode_steps:
                action = self.policy.select_action(state, evaluate=True)

                next_state, reward, done, _ = self.env.step(action)
                t += 1
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= n

        return avg_reward