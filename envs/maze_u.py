import gym
import numpy as np
from envs.maze_env_utils import MazeCell
from envs.maze_task import MazeGoal, MazeTask, BLUE, RED


class GoalRewardUMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.

    def __init__(self, scale):
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0., 0.]) * scale), MazeGoal(np.array([0., 5.]) * scale, rgb=BLUE, deceptive=True)]

    def reward(self, obs):
        if self.termination(obs) == 1:
            return 0.8
        elif self.termination(obs) == 2:
            return 1.
        return self.PENALTY

    @staticmethod
    def create_maze():
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            [B, B, B, B, B, B, B],
            [B, R, R, R, R, R, B],
            [B, B, B, B, B, R, B],
            [B, B, B, B, B, R, B],
            [B, B, B, B, B, R, B],
            [B, B, B, B, B, R, B],
            [B, R, R, R, R, R, B],
            [B, B, B, B, B, B, B],
        ]
