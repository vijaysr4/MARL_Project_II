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


def run_sac_evaluation(mode, opponent_type):
    # 1. Environment Setup
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
    obs, info = env.reset()

    # 2. Dynamic Policy Loading based on Mode
    print(f"Loading SAC Mode: {mode.upper()}...")

    try:
        if mode == "independent":
            # Task 2: Independent (No Shaping)
            base_path = os.path.abspath("./ray_test/sac_v1_independent/iter_49/policies/")
            p0 = Policy.from_checkpoint(os.path.join(base_path, "agent-0-policy"))
            p1 = Policy.from_checkpoint(os.path.join(base_path, "agent-1-policy"))
            p2 = Policy.from_checkpoint(os.path.join(base_path, "agent-2-policy"))
        elif mode == "shared":
            # Task 2: Shared (No Shaping)
            base_path = os.path.abspath("./ray_test/sac_v2_shared/iter_49/policies/")
            master_policy = Policy.from_checkpoint(os.path.join(base_path, "shared-blue-policy"))
            p0 = p1 = p2 = master_policy
        elif mode == "roles":
            # Task 3 Crossover: Independent + Reward Shaping
            base_path = os.path.abspath("./ray_test/sac_v3_custom_roles/iter_49/policies/")
            p0 = Policy.from_checkpoint(os.path.join(base_path, "sac-agent-0-attacker"))
            p1 = Policy.from_checkpoint(os.path.join(base_path, "sac-agent-1-attacker"))
            p2 = Policy.from_checkpoint(os.path.join(base_path, "sac-agent-2-defender"))
    except Exception as e:
        print(f"Error loading policies: {e}. Did you finish training this mode yet?")
        return

    # 3. Load Red Team
    if opponent_type == "heuristic":
        print("Loading MIT Heuristic Red Team (Hard Mode)...")
        red_bot_3 = DefendGen('agent_3', base_env, 'easy')(env.observation_space['agent_3'],
                                                           env.action_space['agent_3'], {})
        red_bot_4 = DefendGen('agent_4', base_env, 'easy')(env.observation_space['agent_4'],
                                                           env.action_space['agent_4'], {})
        red_bot_5 = AttackGen('agent_5', base_env, 'easy')(env.observation_space['agent_5'],
                                                           env.action_space['agent_5'], {})

    # 4. Simulation Loop
    match_data = []
    cum_r0, cum_r1, cum_r2 = 0, 0, 0
    step = 0

    print(f"SIMULATION START: SAC {mode} AI vs {opponent_type.upper()} Bots...")

    try:
        while step < 1000:
            # Blue Actions
            a0 = int(p0.compute_single_action(obs['agent_0'])[0])
            a1 = int(p1.compute_single_action(obs['agent_1'])[0])
            a2 = int(p2.compute_single_action(obs['agent_2'])[0])

            # Red Actions
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

            cum_r0 += reward.get('agent_0', 0)
            cum_r1 += reward.get('agent_1', 0)
            cum_r2 += reward.get('agent_2', 0)

            match_data.append({
                "Step": step,
                "Blue_Score": base_env.state['captures'][0],
                "Red_Score": base_env.state['captures'][1],
                "A0_Reward": cum_r0, "A1_Reward": cum_r1, "A2_Reward": cum_r2
            })

            step += 1
            if any(term.values()) or any(trunc.values()):
                break
    except Exception as e:
        print(f"Match interrupted: {e}")
        traceback.print_exc()
    finally:
        env.close()

    if not match_data: return

    # 5. Plotting
    df = pd.DataFrame(match_data)
    save_dir = f"./plots/SAC_Eval_{mode}_vs_{opponent_type}"
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")

    # Plot Score
    plt.figure(figsize=(8, 4))
    plt.plot(df['Step'], df['Blue_Score'], label='Blue AI', color='blue', lw=3)
    plt.plot(df['Step'], df['Red_Score'], label=f'Red {opponent_type}', color='red', lw=3)
    plt.title(f"SAC {mode.capitalize()} vs {opponent_type.capitalize()}: Score")
    plt.legend();
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scoreboard.png"))

    # Plot Cumulative Rewards (Role Proof)
    plt.figure(figsize=(8, 4))
    plt.plot(df['Step'], df['A0_Reward'], label='Agent 0 (Attacker)', color='crimson')
    plt.plot(df['Step'], df['A1_Reward'], label='Agent 1 (Attacker)', color='orange')
    plt.plot(df['Step'], df['A2_Reward'], label='Agent 2 (Defender)', color='royalblue')
    plt.title(f"SAC {mode.capitalize()} Live Rewards (Role Execution)")
    plt.legend();
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roles.png"))

    print(f"Evaluation Complete. Plots saved to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['independent', 'shared', 'roles'], required=True)
    parser.add_argument('--opponent', type=str, choices=['random', 'heuristic'], default='random')
    args = parser.parse_args()
    run_sac_evaluation(args.mode, args.opponent)