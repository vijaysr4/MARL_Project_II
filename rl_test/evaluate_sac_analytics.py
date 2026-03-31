import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

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


def run_sac_evaluation(paradigm, opponent_type):
    # 1. Environment Setup
    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 5
    config_dict['max_time'] = 240

    base_env = pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict, render_mode=None, reward_config={}, team_size=3)
    env = ParallelPettingZooWrapper(base_env)
    obs, info = env.reset()

    # 2. Load the specified SAC Paradigm
    print(f"Loading SAC Paradigm: {paradigm.upper()}...")

    if paradigm == "independent":
        base_path = os.path.abspath("./ray_test/sac_v1_independent/iter_49/policies/")
        policy_0 = Policy.from_checkpoint(os.path.join(base_path, "agent-0-policy"))
        policy_1 = Policy.from_checkpoint(os.path.join(base_path, "agent-1-policy"))
        policy_2 = Policy.from_checkpoint(os.path.join(base_path, "agent-2-policy"))
    elif paradigm == "shared":
        base_path = os.path.abspath("./ray_test/sac_v2_shared/iter_49/policies/")
        master_policy = Policy.from_checkpoint(os.path.join(base_path, "shared-blue-policy"))
        policy_0 = policy_1 = policy_2 = master_policy

    # 3. Load Red Team
    if opponent_type == "heuristic":
        print("Loading MIT Heuristic Red Team (Hard Mode)...")
        red_bot_3 = DefendGen('agent_3', base_env, 'easy')(env.observation_space['agent_3'],
                                                           env.action_space['agent_3'], {})
        red_bot_4 = DefendGen('agent_4', base_env, 'easy')(env.observation_space['agent_4'],
                                                           env.action_space['agent_4'], {})
        red_bot_5 = AttackGen('agent_5', base_env, 'easy')(env.observation_space['agent_5'],
                                                           env.action_space['agent_5'], {})
    else:
        print("Loading Random Red Team (Evaluation Baseline)...")

    match_data = []
    step = 0

    print(f"SIMULATION START: SAC {paradigm.capitalize()} vs {opponent_type.upper()}...")

    try:
        while step < 1000:
            # Blue Actions (SAC AI)
            a0 = int(policy_0.compute_single_action(obs['agent_0'])[0])
            a1 = int(policy_1.compute_single_action(obs['agent_1'])[0])
            a2 = int(policy_2.compute_single_action(obs['agent_2'])[0])

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

            blue_score = base_env.state['captures'][0]
            red_score = base_env.state['captures'][1]

            match_data.append({"Step": step, "Blue_Score": blue_score, "Red_Score": red_score})

            step += 1
            if any(term.values()) or any(trunc.values()):
                break
    except Exception as e:
        print(f"Match interrupted: {e}")
        traceback.print_exc()
    finally:
        env.close()

    print(
        f"\nMATCH OVER. Final Score -> SAC {paradigm.capitalize()}: {match_data[-1]['Blue_Score']} | Red {opponent_type.capitalize()}: {match_data[-1]['Red_Score']}")

    # Plot the Scoreboard
    if len(match_data) > 0:
        df = pd.DataFrame(match_data)
        save_dir = f"./plots/SAC_{paradigm.capitalize()}_vs_{opponent_type.capitalize()}"
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df, x="Step", y="Blue_Score", color="blue", linewidth=3,
                     label=f"SAC {paradigm.capitalize()} Team")
        sns.lineplot(data=df, x="Step", y="Red_Score", color="red", linewidth=3,
                     label=f"Red Team ({opponent_type.capitalize()})")
        plt.fill_between(df["Step"], df["Blue_Score"], color="blue", alpha=0.1)
        plt.fill_between(df["Step"], df["Red_Score"], color="red", alpha=0.1)

        plt.title(f"SAC {paradigm.capitalize()} Paradigm vs {opponent_type.capitalize()}", fontweight='bold', pad=15)
        plt.xlabel("Match Timesteps", fontweight='bold')
        plt.ylabel("Captures (Points)", fontweight='bold')
        plt.legend(loc="upper left")
        plt.tight_layout()

        plot_path = os.path.join(save_dir, f"sac_{paradigm}_vs_{opponent_type}_scoreboard.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paradigm', type=str, choices=['independent', 'shared'], required=True)
    parser.add_argument('--opponent', type=str, choices=['random', 'heuristic'], default='random')
    args = parser.parse_args()

    run_sac_evaluation(args.paradigm, args.opponent)