import gym
import numpy as np
from envs.maze_env_utils import MazeCell
from envs.maze_task import MazeGoal, MazeTask
from envs.point import PointEnv


class GoalRewardEMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.

    def __init__(self, scale):
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([0., 4.]) * scale)]

    def reward(self, obs):
        return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze():
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        # return [
        #     [B, B, B, B, B, B, B],
        #     [B, R, E, E, E, E, B],
        #     [B, E, E, E, E, E, B],
        #     [B, E, E, E, E, E, B],
        #     [B, E, E, E, E, E, B],
        #     [B, E, E, E, E, E, B],
        #     [B, B, B, B, B, B, B],
        # ]
        return [
            [B, B, B, B, B],
            [B, R, E, E, B],
            [B, B, B, E, B],
            [B, E, E, E, B],
            [B, B, B, E, B],
            [B, E, E, E, B],
            [B, B, B, B, B],
        ]
