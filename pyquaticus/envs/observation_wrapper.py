import gymnasium as gym
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper


class GlobalObservationWrapper(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.possible_agents = env.possible_agents[:]

        obs_dim = self.env.observation_space(self.possible_agents[0]).shape[0]
        global_dim = obs_dim * len(self.possible_agents)

        self.observation_spaces = {
            agent: gym.spaces.Dict({
                "obs": gym.spaces.Box(
                    low=self.env.observation_space(agent).low.astype(np.float32),
                    high=self.env.observation_space(agent).high.astype(np.float32),
                    shape=self.env.observation_space(agent).shape,
                    dtype=np.float32,
                ),
                "global_state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(global_dim,),
                    dtype=np.float32,
                ),
            })
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._add_global_state(obs), infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._add_global_state(obs), rewards, terminations, truncations, infos

    def _add_global_state(self, obs_dict):
        sorted_keys = sorted(obs_dict.keys())
        all_obs = [np.asarray(obs_dict[a], dtype=np.float32) for a in sorted_keys]
        global_state = np.concatenate(all_obs, axis=0).astype(np.float32)

        return {
            agent: {
                "obs": np.asarray(obs_dict[agent], dtype=np.float32),
                "global_state": global_state,
            }
            for agent in obs_dict.keys()
        }