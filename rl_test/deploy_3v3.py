# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# (C) 2023 Massachusetts Institute of Technology.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import gymnasium as gym
import numpy as np
import os
import json  # Added for recording actions

# --- HEADLESS SSH FIX ---
# This must be set before any pygame or pyquaticus imports to bypass the need for a monitor
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import imageio
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPOTF1Policy, PPOTorchPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec
from pyquaticus.base_policies.base_policy_wrappers import DefendGen, AttackGen
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import pyquaticus.utils.rewards as rew

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 3v3 PyQuaticus environment')
    parser.add_argument('policy_one', help='Path to model 1')
    parser.add_argument('policy_two', help='Path to model 2')
    parser.add_argument('policy_three', help='Path to model 3')

    # --- NEW ARGUMENTS FOR FOLDER ORGANIZATION ---
    parser.add_argument('--run_name', help='Name of the model/run (creates a subfolder)', required=True)
    parser.add_argument('--filename', help='Name of the output JSON file', default='replay.json')

    args = parser.parse_args()

    reward_config = {}
    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True

    # 1. Create Environment (render_mode set to None since we are headless)
    base_env = pyquaticus_v0.PyQuaticusEnv(
        config_dict=config_dict,
        render_mode=None,
        reward_config=reward_config,
        team_size=3
    )
    env = ParallelPettingZooWrapper(base_env)

    obs, _ = env.reset()

    # 2. Load Policies
    print("Loading policies...")
    policy_one = Policy.from_checkpoint(os.path.abspath(args.policy_one))
    policy_two = Policy.from_checkpoint(os.path.abspath(args.policy_two))
    policy_three = Policy.from_checkpoint(os.path.abspath(args.policy_three))

    # 3. Setup Action Recording
    action_history = []
    step = 0
    max_step = 1000

    print(f"Simulating and recording actions for {args.run_name}...")

    try:
        while step < max_step:
            # Process Actions for Blue Team
            zero = int(policy_one.compute_single_action(obs['agent_0'])[0])
            one = int(policy_two.compute_single_action(obs['agent_1'])[0])
            two = int(policy_three.compute_single_action(obs['agent_2'])[0])

            # --- CRITICAL FIX: Make the Red Team Move ---
            # Sample random valid actions for the Red Team so they aren't frozen
            three = int(env.action_space['agent_3'].sample())
            four = int(env.action_space['agent_4'].sample())
            five = int(env.action_space['agent_5'].sample())

            actions = {
                'agent_0': zero, 'agent_1': one, 'agent_2': two,
                'agent_3': three, 'agent_4': four, 'agent_5': five
            }

            # Record the step
            action_history.append(actions)

            # Step Env
            obs, reward, term, trunc, info = env.step(actions)

            step += 1
            if step % 100 == 0:
                print(f"Step {step}/{max_step}")

            if any(term.values()) or any(trunc.values()):
                print(f"Match concluded at step {step}.")
                break

    except Exception as e:
        print(f"Error during loop: {e}")
    finally:
        env.close()

        # --- 4. NEW FOLDER/SAVE LOGIC ---
        # Ensure the directory game_vid/<run_name> exists
        save_dir = os.path.join("game_vid", args.run_name)
        os.makedirs(save_dir, exist_ok=True)

        # Ensure the filename ends with .json
        if not args.filename.endswith('.json'):
            args.filename += '.json'

        output_file = os.path.join(save_dir, args.filename)

        with open(output_file, 'w') as f:
            json.dump(action_history, f)

        print(f" SUCCESS: Actions saved to {output_file}")


# python pyquaticus/rl_test/deploy_3v3.py \
#   ./ray_test/baseline_v1_ppo_vs_random/iter_49/policies/agent-0-policy \
#   ./ray_test/baseline_v1_ppo_vs_random/iter_49/policies/agent-1-policy \
#   ./ray_test/baseline_v1_ppo_vs_random/iter_49/policies/agent-2-policy \
#   --run_name "baseline_v1_ppo_vs_random" \
#   --filename "eval_match_1.json"