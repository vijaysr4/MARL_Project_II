# (C) 2021 Massachusetts Institute of Technology.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import gymnasium as gym
import numpy as np
import os
import time
import csv
import logging

import ray
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.logger import UnifiedLogger

from pyquaticus import pyquaticus_v0
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std


class RandPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None,
                        info_batch=None, episodes=None, **kwargs):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self): return {}

    def learn_on_batch(self, samples): return {}

    def set_weights(self, weights): pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    # --- 1. Standard Environment Setup ---
    reward_config = {
        'agent_0': rew.caps_and_grabs, 'agent_1': rew.caps_and_grabs, 'agent_2': rew.caps_and_grabs,
        'agent_3': None, 'agent_4': None, 'agent_5': None
    }

    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True

    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict, render_mode=None,
                                                             reward_config=reward_config, team_size=3)
    env = ParallelPettingZooWrapper(env_creator({}))
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))

    obs_space = env.observation_space['agent_0']
    act_space = env.action_space['agent_0']
    env.close()


    # ==========================================
    # EXPERIMENT 1: SAC INDEPENDENT LEARNING
    # ==========================================
    def policy_mapping_ind(agent_id, episode, worker, **kwargs):
        if agent_id == 'agent_0': return "agent-0-policy"
        if agent_id == 'agent_1': return "agent-1-policy"
        if agent_id == 'agent_2': return "agent-2-policy"
        return "random"


    policies_ind = {
        'agent-0-policy': (None, obs_space, act_space, {}),
        'agent-1-policy': (None, obs_space, act_space, {}),
        'agent-2-policy': (None, obs_space, act_space, {}),
        'random': (RandPolicy, obs_space, act_space, {"no_checkpoint": True})
    }

    config_ind = (SACConfig()
                  .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                  .environment(env='pyquaticus')
                  .env_runners(num_env_runners=1, num_cpus_per_env_runner=1)
                  .training(replay_buffer_config={'type': 'MultiAgentPrioritizedReplayBuffer', 'capacity': 50000})
                  .multi_agent(policies=policies_ind, policy_mapping_fn=policy_mapping_ind,
                               policies_to_train=["agent-0-policy", "agent-1-policy", "agent-2-policy"]))

    RUN_NAME_IND = "sac_v1_independent"
    base_path_ind = f"/users/eleves-b/2024/vijay-venkatesh.murugan/multi_agent_systems/ray_results/{RUN_NAME_IND}"
    os.makedirs(base_path_ind, exist_ok=True)
    algo_ind = config_ind.build_algo(logger_creator=lambda cfg: UnifiedLogger(cfg, base_path_ind, loggers=None))


    # ==========================================
    # EXPERIMENT 2: SAC PARAMETER SHARING
    # ==========================================
    def policy_mapping_shared(agent_id, episode, worker, **kwargs):
        if agent_id in ['agent_0', 'agent_1', 'agent_2']: return "shared-blue-policy"
        return "random"


    policies_shared = {
        'shared-blue-policy': (None, obs_space, act_space, {}),
        'random': (RandPolicy, obs_space, act_space, {"no_checkpoint": True})
    }

    config_shared = (SACConfig()
                     .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                     .environment(env='pyquaticus')
                     .env_runners(num_env_runners=1, num_cpus_per_env_runner=1)
                     .training(replay_buffer_config={'type': 'MultiAgentPrioritizedReplayBuffer', 'capacity': 50000})
                     .multi_agent(policies=policies_shared, policy_mapping_fn=policy_mapping_shared,
                                  policies_to_train=["shared-blue-policy"]))

    RUN_NAME_SHARED = "sac_v2_shared"
    base_path_shared = f"/users/eleves-b/2024/vijay-venkatesh.murugan/multi_agent_systems/ray_results/{RUN_NAME_SHARED}"
    os.makedirs(base_path_shared, exist_ok=True)
    algo_shared = config_shared.build_algo(
        logger_creator=lambda cfg: UnifiedLogger(cfg, base_path_shared, loggers=None))

    # ==========================================
    # CSV SETUP & SIMULTANEOUS TRAINING LOOP
    # ==========================================
    csv_ind = f"{RUN_NAME_IND}_metrics.csv"
    csv_shared = f"{RUN_NAME_SHARED}_metrics.csv"

    # Write Headers with Individual Agents
    for csv_file in [csv_ind, csv_shared]:
        with open(csv_file, mode='w', newline='') as file:
            csv.writer(file).writerow([
                "Iteration", "Mean_Reward", "Episode_Length",
                "Agent_0_Reward", "Agent_1_Reward", "Agent_2_Reward"
            ])

    print("Starting Dual SAC Training: Independent vs Shared Parameters...")

    for i in range(50):
        print(f"\n--- Looping: {i} ---")
        start = time.time()

        # ------------------------------------------------
        # Step 1: Train Independent
        # ------------------------------------------------
        print("Training Independent SAC...")
        res_ind = algo_ind.train()
        mean_rew_ind = res_ind.get('env_runners', {}).get('episode_reward_mean', 0)
        len_ind = res_ind.get('env_runners', {}).get('episode_len_mean', 0)

        # Extract individual policy rewards
        pol_rew_ind = res_ind.get('env_runners', {}).get('policy_reward_mean', {})
        a0_ind = pol_rew_ind.get('agent-0-policy', 0)
        a1_ind = pol_rew_ind.get('agent-1-policy', 0)
        a2_ind = pol_rew_ind.get('agent-2-policy', 0)

        with open(csv_ind, mode='a', newline='') as file:
            csv.writer(file).writerow([i, mean_rew_ind, len_ind, a0_ind, a1_ind, a2_ind])

        # ------------------------------------------------
        # Step 2: Train Shared
        # ------------------------------------------------
        print("Training Shared SAC...")
        res_shared = algo_shared.train()
        mean_rew_shared = res_shared.get('env_runners', {}).get('episode_reward_mean', 0)
        len_shared = res_shared.get('env_runners', {}).get('episode_len_mean', 0)

        # Extract the shared policy reward (applies to all agents)
        pol_rew_shared = res_shared.get('env_runners', {}).get('policy_reward_mean', {})
        shared_pol_rew = pol_rew_shared.get('shared-blue-policy', 0)
        a0_shared = shared_pol_rew
        a1_shared = shared_pol_rew
        a2_shared = shared_pol_rew

        with open(csv_shared, mode='a', newline='') as file:
            csv.writer(file).writerow([i, mean_rew_shared, len_shared, a0_shared, a1_shared, a2_shared])

        end = time.time()
        print(f"Loop {i} Complete in {end - start:.2f} seconds.")
        print(f"Ind Reward: {mean_rew_ind:.2f} | Shared Reward: {mean_rew_shared:.2f}")

        # Save Checkpoints safely
        if np.mod(i, 25) == 0 or i == 49:
            print(f"Saving Checkpoints for Loop {i}...")
            algo_ind.save(f'./ray_test/{RUN_NAME_IND}/iter_{i}/')
            algo_shared.save(f'./ray_test/{RUN_NAME_SHARED}/iter_{i}/')