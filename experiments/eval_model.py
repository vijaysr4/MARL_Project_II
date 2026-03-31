# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
# (C) 2023 Massachusetts Institute of Technology.
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import argparse
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- HEADLESS SSH FIX ---
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import ray
from pyquaticus.envs.observation_wrapper import GlobalObservationWrapper
from ray.rllib.policy.policy import Policy
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus import pyquaticus_v0
from pyquaticus.config import config_dict_std
from pyquaticus.base_policies.base_policy_wrappers import DefendGen, AttackGen


def to_batch(o):
    """Helper to properly batch Dict observations for RLlib heuristic bots."""
    if isinstance(o, dict):
        return {k: [v] for k, v in o.items()}
    return [o]


def run_match_and_plot(args):
    # 1. Standard Environment Setup
    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 5
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True

    base_env = pyquaticus_v0.PyQuaticusEnv(
        config_dict=config_dict, render_mode=None, reward_config={}, team_size=3
    )

    # ==========================================
    # FIX: Add the Global Wrapper for MAPPO
    # ==========================================
    if args.paradigm == "MAPPO":
        base_env = GlobalObservationWrapper(base_env)

    env = ParallelPettingZooWrapper(base_env)

    obs_space = env.observation_space['agent_0']
    act_space = env.action_space['agent_0']
    obs, info = env.reset()

    # 2. Load Blue Team (Dynamic based on Paradigm)
    print(f"Loading Trained Blue Team Policies ({args.paradigm}_{args.reward} - Iter {args.iter})...")
    base_path = os.path.abspath(f"./trained_models/{args.paradigm}_{args.reward}/iter_{args.iter}/policies/")

    blue_policies = {}
    if args.paradigm == "MAPPO":
        # MAPPO uses one shared brain
        shared_pol = Policy.from_checkpoint(os.path.join(base_path, "shared_pol"))
        blue_policies = {0: shared_pol, 1: shared_pol, 2: shared_pol}
    else:
        # IPPO uses three separate brains
        blue_policies = {
            0: Policy.from_checkpoint(os.path.join(base_path, "pol_0")),
            1: Policy.from_checkpoint(os.path.join(base_path, "pol_1")),
            2: Policy.from_checkpoint(os.path.join(base_path, "pol_2"))
        }

    # 3. Load Red Team
    if args.opponent == "heuristic":
        print("Loading MIT Heuristic Red Team (Hard Mode)...")
        red_bot_3 = DefendGen('agent_3', pyquaticus_v0.PyQuaticusEnv(team_size=3), 'easy')(obs_space, act_space, {})
        red_bot_4 = DefendGen('agent_4', pyquaticus_v0.PyQuaticusEnv(team_size=3), 'easy')(obs_space, act_space, {})
        red_bot_5 = AttackGen('agent_5', pyquaticus_v0.PyQuaticusEnv(team_size=3), 'easy')(obs_space, act_space, {})
    else:
        print("Loading Random Red Team (Evaluation Baseline)...")

    # 4. Data Tracking Setup
    match_data = []
    cum_reward_0 = cum_reward_1 = cum_reward_2 = 0
    step = 0
    max_step = 1000

    print(f"\nSIMULATION START: Blue AI vs {args.opponent.upper()} Bots...")

    try:
        while step < max_step:
            # Blue Actions
            # By passing the raw obs directly, RLlib automatically handles both the Dict (MAPPO) and Box (IPPO) spaces.
            a0 = int(blue_policies[0].compute_single_action(obs['agent_0'])[0])
            a1 = int(blue_policies[1].compute_single_action(obs['agent_1'])[0])
            a2 = int(blue_policies[2].compute_single_action(obs['agent_2'])[0])

            # Red Actions
            if args.opponent == "heuristic":
                # 1. Unpack the Suitcase for the Red Team
                obs_3 = obs['agent_3']["obs"] if isinstance(obs['agent_3'], dict) else obs['agent_3']
                obs_4 = obs['agent_4']["obs"] if isinstance(obs['agent_4'], dict) else obs['agent_4']
                obs_5 = obs['agent_5']["obs"] if isinstance(obs['agent_5'], dict) else obs['agent_5']

                # 2. Package the info dictionary EXACTLY as the MIT bots expect it
                # They look for: info[agent_id]["global_state"]
                # Then we wrap THAT inside a batch list.
                info_batch_3 = {'agent_3': [info.get('agent_3', info)]}
                info_batch_4 = {'agent_4': [info.get('agent_4', info)]}
                info_batch_5 = {'agent_5': [info.get('agent_5', info)]}

                # 3. Pass the formatted data
                a3 = int(red_bot_3.compute_actions(obs_batch=[obs_3], info_batch=info_batch_3)[0][0])
                a4 = int(red_bot_4.compute_actions(obs_batch=[obs_4], info_batch=info_batch_4)[0][0])
                a5 = int(red_bot_5.compute_actions(obs_batch=[obs_5], info_batch=info_batch_5)[0][0])
            else:
                a3, a4, a5 = [env.action_space[f'agent_{i}'].sample() for i in range(3, 6)]

            actions = {'agent_0': a0, 'agent_1': a1, 'agent_2': a2, 'agent_3': a3, 'agent_4': a4, 'agent_5': a5}

            obs, reward, term, trunc, info = env.step(actions)

            # Track Cumulative Rewards
            cum_reward_0 += float(reward.get('agent_0', 0))
            cum_reward_1 += float(reward.get('agent_1', 0))
            cum_reward_2 += float(reward.get('agent_2', 0))

            # Use .unwrapped to bypass the wrapper's functions and access the core game dictionary
            blue_score = int(base_env.unwrapped.state['captures'][0])
            red_score = int(base_env.unwrapped.state['captures'][1])

            match_data.append({
                "Step": step,
                "Blue_Score": blue_score,
                "Red_Score": red_score,
                "Attacker_0_CumReward": cum_reward_0,
                "Attacker_1_CumReward": cum_reward_1,
                "Defender_2_CumReward": cum_reward_2
            })

            step += 1
            if any(term.values()) or any(trunc.values()):
                break

    except Exception as e:
        print(f"\nMatch ended or interrupted: {e}")
        traceback.print_exc()
    finally:
        env.close()

    print(f"\nMATCH OVER at step {step}.")
    if len(match_data) > 0:
        print(f"FINAL SCORE - Blue AI: {match_data[-1]['Blue_Score']} | Red {args.opponent.capitalize()}: {match_data[-1]['Red_Score']}")

        # ==========================================
        # SAVING DATA & PLOTTING (Updated to results_eval)
        # ==========================================
        run_identifier = f"{args.paradigm}_{args.reward}_iter{args.iter}_vs_{args.opponent}"
        save_dir = f"./results_eval/{run_identifier}"
        os.makedirs(save_dir, exist_ok=True)

        # 1. Save the JSON Log
        json_path = os.path.join(save_dir, "match_log.json")
        with open(json_path, 'w') as f:
            json.dump(match_data, f, indent=4)
        print(f"SUCCESS: Saved Match JSON log to {json_path}")

        # 2. Save the Plots
        df = pd.DataFrame(match_data)
        sns.set_theme(style="darkgrid", context="paper")
        plt.rcParams.update({"font.family": "serif", "axes.titlesize": 16, "axes.labelsize": 14})

        # PLOT 1: Scoreboard Timeline
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df, x="Step", y="Blue_Score", color="blue", linewidth=3, label="Blue Team (Your AI)")
        sns.lineplot(data=df, x="Step", y="Red_Score", color="red", linewidth=3,
                     label=f"Red Team ({args.opponent.capitalize()})")
        plt.fill_between(df["Step"], df["Blue_Score"], color="blue", alpha=0.1)
        plt.fill_between(df["Step"], df["Red_Score"], color="red", alpha=0.1)
        plt.title(f"Scoreboard: {args.paradigm} vs {args.opponent.capitalize()}", fontweight='bold', pad=15)
        plt.xlabel("Match Timesteps", fontweight='bold')
        plt.ylabel("Captures (Points)", fontweight='bold')
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "01_match_score_timeline.png"), dpi=300)
        plt.close()

        # PLOT 2: Live In-Game Role Execution
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df, x="Step", y="Attacker_0_CumReward", color="crimson", linewidth=2.5,
                     label="Agent 0 (Attacker)")
        sns.lineplot(data=df, x="Step", y="Attacker_1_CumReward", color="darkorange", linewidth=2.5, linestyle="--",
                     label="Agent 1 (Attacker)")
        sns.lineplot(data=df, x="Step", y="Defender_2_CumReward", color="royalblue", linewidth=2.5,
                     label="Agent 2 (Defender)")
        plt.title(f"Role Rewards: {args.paradigm} vs {args.opponent.capitalize()}", fontweight='bold', pad=15)
        plt.xlabel("Match Timesteps", fontweight='bold')
        plt.ylabel("Cumulative In-Game Reward", fontweight='bold')
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "02_match_role_execution.png"), dpi=300)
        plt.close()

        print(f"SUCCESS: Saved Match Analytics plots to {save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Blue Team AI.')
    parser.add_argument('--paradigm', type=str, choices=['IPPO', 'MAPPO'], required=True)
    parser.add_argument('--reward', type=str, choices=['SPARSE', 'SHAPED'], required=True)
    parser.add_argument('--iter', type=int, default=50, help='Which iteration checkpoint to load (default: 50)')
    parser.add_argument('--opponent', type=str, choices=['random', 'heuristic'], default='heuristic')
    args = parser.parse_args()

    # Initialize Ray for policy loading
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    # ==========================================
    # FIX: Register the Custom Neural Network Blueprint
    # ==========================================
    from ray.rllib.models import ModelCatalog
    from pyquaticus.models.marl_models import CentralizedCriticModel
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    run_match_and_plot(args)
    ray.shutdown()