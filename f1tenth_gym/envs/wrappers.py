"""Gymnasium wrappers offering clean interfaces onto the multi-agent F110Env.

These are deliberately thin adapters -- the native env keeps its multi-agent
``dict[agent -> dict[field -> value]]`` observation (load-bearing for numba
consumers). Use these to consume it from single-agent RL code.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

__all__ = ["SingleAgentWrapper"]


class SingleAgentWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Flat single-agent interface over a 1-agent F110Env.

    Unwraps ``obs["agent_0"]`` to the observation and reshapes the action from
    ``(1, 2)`` to ``(2,)``, so the env is directly consumable by single-agent RL
    libraries (SB3, CleanRL, ...). Compose with
    ``gymnasium.wrappers.FlattenObservation`` for a flat ``Box`` observation.

    Requires ``num_agents == 1``. Reward is already a scalar; ``info`` is passed
    through unchanged (its per-agent arrays have length 1).
    """

    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        num_agents = env.unwrapped.num_agents
        if num_agents != 1:
            raise ValueError(
                f"SingleAgentWrapper requires num_agents == 1, got {num_agents}"
            )
        self._agent_id = env.unwrapped.agent_ids[0]
        self.observation_space = env.observation_space[self._agent_id]
        act = env.action_space  # Box(shape=(1, 2))
        self.action_space = gym.spaces.Box(
            low=np.asarray(act.low[0]), high=np.asarray(act.high[0]), dtype=act.dtype
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[self._agent_id], info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs[self._agent_id], reward, terminated, truncated, info
