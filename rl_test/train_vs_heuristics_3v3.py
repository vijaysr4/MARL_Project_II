# (C) 2021 Massachusetts Institute of Technology.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import gymnasium as gym
import numpy as np
import pygame
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec, Policy
import os
import pyquaticus.utils.rewards as rew
from pyquaticus.config import config_dict_std
import logging
import csv


class RandPolicy(Policy):
    """
    Wrapper for training against a random policy.
    This guarantees the bots will move unpredictably without crashing the RolloutWorker.
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self): return {}

    def learn_on_batch(self, samples): return {}

    def set_weights(self, weights): pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 3v3 policy in a 3v3 PyQuaticus environment')
    parser.add_argument('--render', help='Enable rendering', action='store_true')

    # Example Reward Config
    reward_config = {
        'agent_0': rew.caps_and_grabs,
        'agent_1': rew.caps_and_grabs,
        'agent_2': rew.caps_and_grabs,
        'agent_3': None,
        'agent_4': None,
        'agent_5': None
    }

    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    RENDER_MODE = 'human' if args.render else None

    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True

    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(
        config_dict=config_dict, render_mode=RENDER_MODE, reward_config=reward_config, team_size=3
    )
    env = ParallelPettingZooWrapper(
        pyquaticus_v0.PyQuaticusEnv(
            config_dict=config_dict, render_mode=RENDER_MODE, reward_config=reward_config, team_size=3
        )
    )
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    obs_space = env.observation_space['agent_0']
    act_space = env.action_space['agent_0']


    # --- CHANGED: Map Red Team to Random Moving Bots ---
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # Blue Team (Learning AI)
        if agent_id == 'agent_0': return "agent-0-policy"
        if agent_id == 'agent_1': return "agent-1-policy"
        if agent_id == 'agent_2': return "agent-2-policy"

        # Red Team mapped to the safe random policy
        if agent_id in ['agent_3', 'agent_4', 'agent_5']: return "random"
        return "random"


    # --- CHANGED: Safely assign the random policy ---
    policies = {
        'agent-0-policy': (None, obs_space, act_space, {}),
        'agent-1-policy': (None, obs_space, act_space, {}),
        'agent-2-policy': (None, obs_space, act_space, {}),
        'random': (RandPolicy, obs_space, act_space, {"no_checkpoint": True})
    }

    env.close()

    ppo_config = (PPOConfig()
                  .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                  .environment(env='pyquaticus').env_runners(num_env_runners=1, num_cpus_per_env_runner=1))

    ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn,
                           policies_to_train=["agent-0-policy", "agent-1-policy", "agent-2-policy"], )

    # --- CHANGED: Run Name ---
    RUN_NAME = "baseline_v2_vs_random"
    base_path = f"/users/eleves-b/2024/vijay-venkatesh.murugan/multi_agent_systems/ray_results/{RUN_NAME}"

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    from ray.tune.logger import UnifiedLogger


    def custom_logger_creator(config):
        return UnifiedLogger(config, base_path, loggers=None)


    algo = ppo_config.build_algo(logger_creator=custom_logger_creator)

    csv_filename = f"{RUN_NAME}_metrics.csv"

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Iteration", "Mean_Reward", "Episode_Length",
            "Agent_0_Reward", "Agent_1_Reward", "Agent_2_Reward"
        ])

    start = 0
    end = 0
    for i in range(50):
        print("Looping: ", i)
        start = time.time()
        result = algo.train()
        end = time.time()
        print("End Loop: ", end - start)

        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', result.get('episode_reward_mean', 0))
        episode_len = result.get('env_runners', {}).get('episode_len_mean', result.get('episode_len_mean', 0))

        policy_rewards = result.get('env_runners', {}).get('policy_reward_mean', result.get('policy_reward_mean', {}))
        agent_0_rew = policy_rewards.get('agent-0-policy', 0)
        agent_1_rew = policy_rewards.get('agent-1-policy', 0)
        agent_2_rew = policy_rewards.get('agent-2-policy', 0)

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, mean_reward, episode_len, agent_0_rew, agent_1_rew, agent_2_rew])

        if np.mod(i, 25) == 0 or i == 49:
            print("Saving Checkpoint: ", i)
            chkpt_file = algo.save(f'./ray_test/{RUN_NAME}/iter_{i}/')