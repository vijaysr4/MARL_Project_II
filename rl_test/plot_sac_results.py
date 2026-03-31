import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def generate_sac_plots(ind_csv, shared_csv):
    # 1. Create the Directory Structure
    base_plot_dir = "./plots"
    save_dir = os.path.join(base_plot_dir, "SAC_Comparison")
    os.makedirs(save_dir, exist_ok=True)

    # 2. Load the Data
    try:
        df_ind = pd.read_csv(ind_csv)
        df_shared = pd.read_csv(shared_csv)
    except FileNotFoundError as e:
        print(f"Error: Could not find the CSV files. Make sure you are in the right folder!\nDetails: {e}")
        return

    # 3. Set "Fake" LaTeX Aesthetics
    sns.set_theme(style="darkgrid", context="paper")
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    # Create a dynamic rolling window to smooth the spiky RL data
    window_size = max(1, len(df_ind) // 10)

    # Apply smoothing to all metrics
    for df in [df_ind, df_shared]:
        df['Mean_Reward_Smooth'] = df['Mean_Reward'].rolling(window=window_size, min_periods=1).mean()
        df['Episode_Length_Smooth'] = df['Episode_Length'].rolling(window=window_size, min_periods=1).mean()
        for i in range(3):
            df[f'Agent_{i}_Smooth'] = df[f'Agent_{i}_Reward'].rolling(window=window_size, min_periods=1).mean()

    print(f"Generating academic-style comparison plots in {save_dir}...")

    # --- PLOT 1: MARL Paradigm Comparison (Team Reward) ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_ind, x="Iteration", y="Mean_Reward", color="lightblue", alpha=0.3)
    sns.lineplot(data=df_ind, x="Iteration", y="Mean_Reward_Smooth", color="navy", linewidth=2.5,
                 label="Independent SAC (3 Brains)")

    sns.lineplot(data=df_shared, x="Iteration", y="Mean_Reward", color="lightcoral", alpha=0.3)
    sns.lineplot(data=df_shared, x="Iteration", y="Mean_Reward_Smooth", color="darkred", linewidth=2.5,
                 label="Shared SAC (1 Master Brain)")

    plt.title("Task 2: MARL Paradigm Reward Comparison", fontweight='bold', pad=15)
    plt.xlabel("Training Iterations", fontweight='bold')
    plt.ylabel("Mean Episode Reward", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_sac_team_reward_comparison.png"), dpi=300)
    plt.close()

    # --- PLOT 2: Tactical Efficiency (Episode Length) ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_ind, x="Iteration", y="Episode_Length", color="thistle", alpha=0.3)
    sns.lineplot(data=df_ind, x="Iteration", y="Episode_Length_Smooth", color="purple", linewidth=2.5,
                 label="Independent SAC")

    sns.lineplot(data=df_shared, x="Iteration", y="Episode_Length", color="palegreen", alpha=0.3)
    sns.lineplot(data=df_shared, x="Iteration", y="Episode_Length_Smooth", color="darkgreen", linewidth=2.5,
                 label="Shared SAC")

    plt.title("Task 2: Tactical Efficiency Comparison", fontweight='bold', pad=15)
    plt.xlabel("Training Iterations", fontweight='bold')
    plt.ylabel("Average Steps per Match", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "02_sac_episode_length_comparison.png"), dpi=300)
    plt.close()

    # --- PLOT 3: Individual Agents - Independent SAC ---
    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("deep")
    for i in range(3):
        sns.lineplot(data=df_ind, x="Iteration", y=f"Agent_{i}_Smooth", label=f"Agent {i}", color=colors[i],
                     linewidth=2)

    plt.title("Independent SAC: Individual Agent Performance", fontweight='bold', pad=15)
    plt.xlabel("Training Iterations", fontweight='bold')
    plt.ylabel("Smoothed Reward", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_sac_ind_agents.png"), dpi=300)
    plt.close()

    # --- PLOT 4: Individual Agents - Shared SAC ---
    # Note: Because they share a brain, their rewards are perfectly synced.
    # We use different line thicknesses so you can visually see they overlap.
    plt.figure(figsize=(8, 5))
    for i in range(3):
        sns.lineplot(data=df_shared, x="Iteration", y=f"Agent_{i}_Smooth", label=f"Agent {i}", color=colors[i],
                     linewidth=4 - i, alpha=0.8)

    plt.title("Shared SAC: Individual Agent Performance (Clones)", fontweight='bold', pad=15)
    plt.xlabel("Training Iterations", fontweight='bold')
    plt.ylabel("Smoothed Reward", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "04_sac_shared_agents.png"), dpi=300)
    plt.close()

    print(f"SUCCESS: Saved 4 high-resolution plots to {save_dir}/")


if __name__ == '__main__':
    # Hardcoding the exact outputs from the dual SAC script
    ind_csv_file = "sac_v1_independent_metrics.csv"
    shared_csv_file = "sac_v2_shared_metrics.csv"

    generate_sac_plots(ind_csv_file, shared_csv_file)