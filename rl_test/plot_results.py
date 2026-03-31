import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse


def generate_plots(csv_filename, run_name):
    base_plot_dir = "./plots"
    save_dir = os.path.join(base_plot_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filename}. Verify the path!")
        return

    sns.set_theme(style="darkgrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "font.weight": "bold"
    })

    # Smoothing Window
    window_size = 5
    df['Mean_Reward_Smooth'] = df['Mean_Reward'].rolling(window=window_size, min_periods=1).mean()
    df['Episode_Length_Smooth'] = df['Episode_Length'].rolling(window=window_size, min_periods=1).mean()

    print(f"Generating Heuristic-Comparison plots for {run_name}...")

    # --- PLOT 1: Team Learning Curve ---
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=df, x="Iteration", y="Mean_Reward", color="lightblue", alpha=0.3, label="Raw Data")
    sns.lineplot(data=df, x="Iteration", y="Mean_Reward_Smooth", color="navy", linewidth=2.5, label="Smoothed Trend")
    plt.title(f"Team Learning Curve: {run_name}", fontweight='bold', pad=15)
    plt.xlabel("Training Iterations")
    plt.ylabel("Mean Episode Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_team_reward.png"), dpi=300)

    # --- PLOT 2: Tactical Efficiency (Survival Time) ---
    # CRITICAL: This shows how long you lasted against the MIT Heuristics!
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=df, x="Iteration", y="Episode_Length", color="thistle", alpha=0.3)
    sns.lineplot(data=df, x="Iteration", y="Episode_Length_Smooth", color="purple", linewidth=2.5,
                 label="Avg Survival (Steps)")
    plt.title(f"Tactical Efficiency: {run_name}", fontweight='bold', pad=15)
    plt.xlabel("Training Iterations")
    plt.ylabel("Steps per Match (Max 600)")
    plt.axhline(y=600, color='r', linestyle='--', alpha=0.5, label="Time Limit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "02_tactical_efficiency.png"), dpi=300)

    # --- PLOT 3: Individual Contribution ---
    plt.figure(figsize=(9, 6))
    colors = ["#e74c3c", "#f39c12", "#3498db"]  # Attacker 1, Attacker 2, Defender
    labels = ["Agent 0 (Attacker)", "Agent 1 (Attacker)", "Agent 2 (Defender)"]

    for i in range(3):
        col = f'Agent_{i}_Reward'
        smooth_col = f'Agent_{i}_Smooth'
        df[smooth_col] = df[col].rolling(window=window_size, min_periods=1).mean()
        sns.lineplot(data=df, x="Iteration", y=smooth_col, label=labels[i], color=colors[i], linewidth=2)

    plt.title(f"Individual Agent Specialization: {run_name}", fontweight='bold', pad=15)
    plt.xlabel("Training Iterations")
    plt.ylabel("Smoothed Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_individual_performance.png"), dpi=300)
    plt.close('all')

    print(f"SUCCESS: Saved plots to {save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('run_name')
    args = parser.parse_args()
    generate_plots(args.csv_file, args.run_name)