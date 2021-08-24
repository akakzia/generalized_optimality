import gym
import numpy as np
from envs.maze_env_utils import MazeCell
from envs.maze_task import MazeGoal, MazeTask, BLUE, RED
from envs.point import PointEnv


class GoalRewardDeceptiveMaze(MazeTask):
    REWARD_THRESHOLD: float = 0.9
    PENALTY: float = -0.

    def __init__(self, scale):
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([-4., 9.]) * scale), MazeGoal(np.array([0., 0.]) * scale, rgb=BLUE, deceptive=True)]

    def reward(self, obs):
        if self.termination(obs) == 1:
            return 0.5
        elif self.termination(obs) == 2:
            return 1.
        return self.PENALTY
        # return 1.0 if self.termination(obs) else self.PENALTY

    @staticmethod
    def create_maze():
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
        return [
            # [B, B, B, B, B, B],
            # [B, E, E, E, E, B],
            # [B, E, R, E, E, B],
            # [B, E, E, E, E, B],
            # [B, B, E, E, E, B],
            # [B, E, E, E, E, B],
            # [B, B, B, B, B, B],
            # [B, B, B, B, B, B],
            # [B, R, E, E, B, B],
            # [B, B, B, E, B, B],
            # [B, B, B, E, E, B],
            # [B, B, B, E, B, B],
            # [B, E, E, E, B, B],
            # [B, B, B, B, B, B],
            # [B, B, B, B, B, B],
            # [B, E, R, R, R, B],
            # [B, B, B, B, R, B],
            # [B, B, B, B, R, B],
            # [B, B, B, B, R, B],
            # [B, E, R, R, R, B],
            # [B, B, B, B, B, B],
            # [B, B, B, B, B, B, B, B, B, B, B, B],
            # [B, R, R, R, R, B, B, B, B, B, B, B],
            # [B, B, B, B, R, B, B, B, B, B, B, B],
            # [B, B, B, B, R, B, B, B, B, B, B, B],
            # [B, B, B, B, R, B, B, B, B, B, B, B],
            # [B, B, B, B, R, R, R, R, R, R, R, B],
            # [B, B, B, B, B, B, B, B, B, B, B, B],
            [B, B, B, B, B, B, B],
            [B, B, B, B, B, R, B],
            [B, B, B, B, B, R, B],
            [B, B, B, B, B, R, B],
            [B, B, B, B, B, R, B],
            [B, R, R, R, R, R, B],
            [B, R, B, B, B, B, B],
            [B, R, B, B, B, B, B],
            [B, R, B, B, B, B, B],
            [B, R, B, B, B, B, B],
            [B, R, B, B, B, B, B],
            [B, B, B, B, B, B, B],
        ]
