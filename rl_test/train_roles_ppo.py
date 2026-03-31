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
from ray.rllib.algorithms.ppo import PPOConfig
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

    # ==========================================
    # TASK 3: CUSTOM REWARD ROLES
    # ==========================================
    # Agent 0 and 1 are rewarded for rushing the enemy flag.
    # Agent 2 is rewarded for staying home and defending.
    reward_config = {
        'agent_0': rew.pure_attacker,
        'agent_1': rew.pure_attacker,
        'agent_2': rew.pure_defender,
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


    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # We give them specific names so we can track their unique brains
        if agent_id == 'agent_0': return "agent-0-attacker"
        if agent_id == 'agent_1': return "agent-1-attacker"
        if agent_id == 'agent_2': return "agent-2-defender"
        return "random"


    policies = {
        'agent-0-attacker': (None, obs_space, act_space, {}),
        'agent-1-attacker': (None, obs_space, act_space, {}),
        'agent-2-defender': (None, obs_space, act_space, {}),
        'random': (RandPolicy, obs_space, act_space, {"no_checkpoint": True})
    }

    ppo_config = (PPOConfig()
                  .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                  .environment(env='pyquaticus').env_runners(num_env_runners=1, num_cpus_per_env_runner=1)
                  .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn,
                               policies_to_train=["agent-0-attacker", "agent-1-attacker", "agent-2-defender"]))

    RUN_NAME = "ppo_v3_custom_roles"
    base_path = f"/users/eleves-b/2024/vijay-venkatesh.murugan/multi_agent_systems/ray_results/{RUN_NAME}"
    os.makedirs(base_path, exist_ok=True)
    algo = ppo_config.build_algo(logger_creator=lambda cfg: UnifiedLogger(cfg, base_path, loggers=None))

    # --- CSV Setup ---
    csv_filename = f"{RUN_NAME}_metrics.csv"
    with open(csv_filename, mode='w', newline='') as file:
        csv.writer(file).writerow(
            ["Iteration", "Mean_Reward", "Episode_Length", "Agent_0_Reward", "Agent_1_Reward", "Agent_2_Reward"])

    print("Starting PPO Training with Custom Roles (Attacker vs Defender)...")

    for i in range(50):
        print(f"\n--- Looping: {i} ---")
        start = time.time()
        result = algo.train()
        end = time.time()

        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', 0)
        episode_len = result.get('env_runners', {}).get('episode_len_mean', 0)

        # Extract specific rewards for our newly shaped roles
        policy_rewards = result.get('env_runners', {}).get('policy_reward_mean', {})
        a0_att_rew = policy_rewards.get('agent-0-attacker', 0)
        a1_att_rew = policy_rewards.get('agent-1-attacker', 0)
        a2_def_rew = policy_rewards.get('agent-2-defender', 0)

        print(f"Loop Complete: {end - start:.2f}s")
        print(
            f"Team Mean: {mean_reward:.2f} | Attacker 0: {a0_att_rew:.2f} | Attacker 1: {a1_att_rew:.2f} | Defender 2: {a2_def_rew:.2f}")

        with open(csv_filename, mode='a', newline='') as file:
            csv.writer(file).writerow([i, mean_reward, episode_len, a0_att_rew, a1_att_rew, a2_def_rew])

        # Save Checkpoint
        if np.mod(i, 25) == 0 or i == 49:
            print(f"Saving Checkpoint: {i}")
            algo.save(f'./ray_test/{RUN_NAME}/iter_{i}/')