import argparse
import gymnasium as gym
import numpy as np
import os
import time
import csv
import logging
import torch
import torch.nn as nn

import ray
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger
from pettingzoo.utils.wrappers import BaseParallelWrapper

from pyquaticus import pyquaticus_v0
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std


# ==========================================
# 1. THE CENTRALIZED CRITIC MODEL
# ==========================================
class CentralizedCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # RLlib wraps Dict spaces in a special way.
        # We access the underlying spaces using .original_space
        internal_obs_space = obs_space.original_space["obs"]
        internal_global_space = obs_space.original_space["global_state"]

        # Actor: only sees local 'obs'
        self.actor = TorchFC(
            internal_obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_actor"
        )

        # Critic: sees 'global_state'
        self.critic_net = nn.Sequential(
            nn.Linear(internal_global_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self._global_obs = None

    def forward(self, input_dict, state, seq_lens):
        # Store global state for the value function call
        self._global_obs = input_dict["obs"]["global_state"]
        # Use local obs for the actor
        return self.actor({"obs": input_dict["obs"]["obs"]}, state, seq_lens)

    def value_function(self):
        return torch.reshape(self.critic_net(self._global_obs), [-1])


# ==========================================
# 2. THE OBSERVATION WRAPPER (MAPPO Needs this)
# ==========================================
class GlobalObservationWrapper(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.possible_agents = env.possible_agents[:]
        obs_dim = self.env.observation_space(self.possible_agents[0]).shape[0]
        global_dim = obs_dim * len(self.possible_agents)

        self.observation_spaces = {
            agent: gym.spaces.Dict({
                "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                "global_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(global_dim,), dtype=np.float32),
            }) for agent in self.possible_agents
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
        # Concatenate all agent observations into one vector
        all_obs = [np.asarray(obs_dict[a], dtype=np.float32) for a in self.possible_agents]
        global_state = np.concatenate(all_obs, axis=0)
        return {a: {"obs": np.asarray(obs_dict[a], dtype=np.float32), "global_state": global_state} for a in
                obs_dict.keys()}


# ==========================================
# 3. TRAINING EXECUTION
# ==========================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    ray.init(ignore_reinit_error=True)

    ModelCatalog.register_custom_model("mappo_model", CentralizedCriticModel)

    reward_config = {
        'agent_0': rew.pure_attacker, 'agent_1': rew.pure_attacker, 'agent_2': rew.pure_defender,
        'agent_3': None, 'agent_4': None, 'agent_5': None
    }

    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4


    def env_creator(config):
        env = pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict, reward_config=reward_config, team_size=3)
        return ParallelPettingZooWrapper(GlobalObservationWrapper(env))


    register_env('pyquaticus_mappo', env_creator)

    # Get spaces for policy init
    test_env = env_creator({})
    obs_space = test_env.observation_space['agent_0']
    act_space = test_env.action_space['agent_0']
    test_env.close()

    policies = {
        "shared_mappo_policy": (None, obs_space, act_space, {"model": {"custom_model": "mappo_model"}}),
        "random": (None, obs_space, act_space, {})  # Simplified for testing
    }


    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "shared_mappo_policy" if int(agent_id[-1]) < 3 else "random"


    ppo_config = (PPOConfig()
                  .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                  .environment(env='pyquaticus_mappo')
                  .env_runners(num_env_runners=1)
                  .multi_agent(policies=policies,
                               policy_mapping_fn=policy_mapping_fn,
                               policies_to_train=["shared_mappo_policy"]))

    RUN_NAME = "mappo_central_critic"
    base_path = os.path.abspath(f"./ray_results/{RUN_NAME}")
    os.makedirs(base_path, exist_ok=True)

    algo = ppo_config.build_algo(logger_creator=lambda cfg: UnifiedLogger(cfg, base_path, loggers=None))

    print("Starting MAPPO Centralized Critic Training...")

    for i in range(100):
        result = algo.train()
        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', 0)
        print(f"Iteration: {i} | Mean Reward: {mean_reward:.2f}")

        if i % 10 == 0:
            algo.save(f"./checkpoints/{RUN_NAME}/iter_{i}")