"""
Mujoco Maze
----------

A maze environment using mujoco that supports custom tasks and robots.
"""


import gym

from envs.ant import AntEnv
from envs.maze_task import TaskRegistry
from envs.point import PointEnv
from envs.reacher import ReacherEnv
from envs.swimmer import SwimmerEnv
from envs.maze_custom import GoalRewardEMaze
from envs.maze_deceptive import GoalRewardDeceptiveMaze

for maze_id in TaskRegistry.keys():
    for i, task_cls in enumerate(TaskRegistry.tasks(maze_id)):
        point_scale = task_cls.MAZE_SIZE_SCALING.point
        if point_scale is not None:
            # Point
            gym.envs.register(
                id=f"Point{maze_id}-v{i}",
                entry_point="envs.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=PointEnv,
                    maze_task=task_cls,
                    maze_size_scaling=point_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=1000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )

        ant_scale = task_cls.MAZE_SIZE_SCALING.ant
        if ant_scale is not None:
            # Ant
            gym.envs.register(
                id=f"Ant{maze_id}-v{i}",
                entry_point="envs.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=AntEnv,
                    maze_task=task_cls,
                    maze_size_scaling=ant_scale,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=1000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )

        swimmer_scale = task_cls.MAZE_SIZE_SCALING.swimmer
        if swimmer_scale is not None:
            # Reacher
            gym.envs.register(
                id=f"Reacher{maze_id}-v{i}",
                entry_point="envs.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=ReacherEnv,
                    maze_task=task_cls,
                    maze_size_scaling=task_cls.MAZE_SIZE_SCALING.swimmer,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=1000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )
            # Swimmer
            gym.envs.register(
                id=f"Swimmer{maze_id}-v{i}",
                entry_point="envs.maze_env:MazeEnv",
                kwargs=dict(
                    model_cls=SwimmerEnv,
                    maze_task=task_cls,
                    maze_size_scaling=task_cls.MAZE_SIZE_SCALING.swimmer,
                    inner_reward_scaling=task_cls.INNER_REWARD_SCALING,
                ),
                max_episode_steps=1000,
                reward_threshold=task_cls.REWARD_THRESHOLD,
            )


gym.envs.register(
    id="CustomSquare-v0",
    entry_point="envs.maze_env:MazeEnv",
    kwargs=dict(
        model_cls=PointEnv,
        maze_task=GoalRewardEMaze,
        maze_size_scaling=GoalRewardEMaze.MAZE_SIZE_SCALING.point,
        inner_reward_scaling=GoalRewardEMaze.INNER_REWARD_SCALING,
    )
)

gym.envs.register(
    id="DeceptiveSquare-v0",
    entry_point="envs.maze_env:MazeEnv",
    kwargs=dict(
        model_cls=PointEnv,
        maze_task=GoalRewardDeceptiveMaze,
        maze_size_scaling=GoalRewardDeceptiveMaze.MAZE_SIZE_SCALING.point,
        inner_reward_scaling=GoalRewardDeceptiveMaze.INNER_REWARD_SCALING,
    )
)
__version__ = "0.2.0"
