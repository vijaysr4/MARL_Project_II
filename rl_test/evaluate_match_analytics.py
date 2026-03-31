# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
# (C) 2023 Massachusetts Institute of Technology.
# SPDX-License-Identifier: BSD-3-Clause

import os
import argparse
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- HEADLESS SSH FIX ---
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import ray
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


def run_match_and_plot(opponent_type):
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
    env = ParallelPettingZooWrapper(base_env)

    obs_space = env.observation_space['agent_0']
    act_space = env.action_space['agent_0']
    obs, info = env.reset()

    # 2. Load Blue Team (Your Task 3 Custom Roles AI)
    print(f"Loading Trained Blue Team Policies (Custom Roles)...")
    base_path = os.path.abspath("./ray_test/ppo_v3_custom_roles/iter_49/policies/")
    policy_0 = Policy.from_checkpoint(os.path.join(base_path, "agent-0-attacker"))
    policy_1 = Policy.from_checkpoint(os.path.join(base_path, "agent-1-attacker"))
    policy_2 = Policy.from_checkpoint(os.path.join(base_path, "agent-2-defender"))

    # 3. Load Red Team (If Heuristic)
    if opponent_type == "heuristic":
        print("Loading MIT Heuristic Red Team (Hard Mode)...")
        red_bot_3 = DefendGen('agent_3', base_env, 'easy')(obs_space, act_space, {})
        red_bot_4 = DefendGen('agent_4', base_env, 'easy')(obs_space, act_space, {})
        red_bot_5 = AttackGen('agent_5', base_env, 'easy')(obs_space, act_space, {})
    else:
        print("Loading Random Red Team (Evaluation Baseline)...")

    # 4. Data Tracking Setup
    match_data = []
    cum_reward_0 = 0
    cum_reward_1 = 0
    cum_reward_2 = 0

    step = 0
    max_step = 1000

    print(f"\nSIMULATION START: Blue AI vs {opponent_type.upper()} Bots...")

    try:
        while step < max_step:
            # Blue Actions (Your Trained AI)
            a0 = int(policy_0.compute_single_action(obs['agent_0'])[0])
            a1 = int(policy_1.compute_single_action(obs['agent_1'])[0])
            a2 = int(policy_2.compute_single_action(obs['agent_2'])[0])

            # Red Actions (Switch based on opponent type)
            if opponent_type == "heuristic":
                a3 = int(red_bot_3.compute_actions(obs_batch=to_batch(obs['agent_3']), info_batch=to_batch(info))[0][0])
                a4 = int(red_bot_4.compute_actions(obs_batch=to_batch(obs['agent_4']), info_batch=to_batch(info))[0][0])
                a5 = int(red_bot_5.compute_actions(obs_batch=to_batch(obs['agent_5']), info_batch=to_batch(info))[0][0])
            else:
                a3 = env.action_space['agent_3'].sample()
                a4 = env.action_space['agent_4'].sample()
                a5 = env.action_space['agent_5'].sample()

            actions = {'agent_0': a0, 'agent_1': a1, 'agent_2': a2, 'agent_3': a3, 'agent_4': a4, 'agent_5': a5}

            obs, reward, term, trunc, info = env.step(actions)

            # Track Cumulative Rewards
            cum_reward_0 += reward.get('agent_0', 0)
            cum_reward_1 += reward.get('agent_1', 0)
            cum_reward_2 += reward.get('agent_2', 0)

            blue_score = base_env.state['captures'][0]
            red_score = base_env.state['captures'][1]

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
        print(
            f"FINAL SCORE - Blue AI: {match_data[-1]['Blue_Score']} | Red {opponent_type.capitalize()}: {match_data[-1]['Red_Score']}")

        # ==========================================
        # PLOTTING THE MATCH ANALYTICS
        # ==========================================
        df = pd.DataFrame(match_data)

        # Save to different folders based on opponent
        save_dir = f"./plots/Match_Analytics_vs_{opponent_type.capitalize()}"
        os.makedirs(save_dir, exist_ok=True)

        sns.set_theme(style="darkgrid", context="paper")
        plt.rcParams.update({"font.family": "serif", "axes.titlesize": 16, "axes.labelsize": 14})

        # PLOT 1: Scoreboard Timeline
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df, x="Step", y="Blue_Score", color="blue", linewidth=3, label="Blue Team (Your AI)")
        sns.lineplot(data=df, x="Step", y="Red_Score", color="red", linewidth=3,
                     label=f"Red Team ({opponent_type.capitalize()})")

        plt.fill_between(df["Step"], df["Blue_Score"], color="blue", alpha=0.1)
        plt.fill_between(df["Step"], df["Red_Score"], color="red", alpha=0.1)

        plt.title(f"Live Scoreboard vs {opponent_type.capitalize()} Opponent", fontweight='bold', pad=15)
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

        plt.title(f"Live Role Execution vs {opponent_type.capitalize()} Opponent", fontweight='bold', pad=15)
        plt.xlabel("Match Timesteps", fontweight='bold')
        plt.ylabel("Cumulative In-Game Reward", fontweight='bold')
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "02_match_role_execution.png"), dpi=300)
        plt.close()

        print(f"SUCCESS: Saved Match Analytics plots to {save_dir}/")
    else:
        print("Not enough match data was generated to plot graphs.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Blue Team AI against different opponents.')
    parser.add_argument('--opponent', type=str, choices=['random', 'heuristic'], default='random',
                        help='Choose the Red Team opponent: "random" or "heuristic". Default is "random".')
    args = parser.parse_args()

    run_match_and_plot(args.opponent)