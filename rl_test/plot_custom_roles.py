import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def generate_role_plots(csv_filename):
    # 1. Create the Directory Structure
    base_plot_dir = "./plots"
    save_dir = os.path.join(base_plot_dir, "PPO_Custom_Roles")
    os.makedirs(save_dir, exist_ok=True)

    # 2. Load the Data
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError as e:
        print(f"Error: Could not find {csv_filename}. Make sure it is in the same folder!\nDetails: {e}")
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
    window_size = max(1, len(df) // 10)

    # Apply smoothing
    df['Mean_Reward_Smooth'] = df['Mean_Reward'].rolling(window=window_size, min_periods=1).mean()
    df['Episode_Length_Smooth'] = df['Episode_Length'].rolling(window=window_size, min_periods=1).mean()
    df['Attacker_0_Smooth'] = df['Agent_0_Reward'].rolling(window=window_size, min_periods=1).mean()
    df['Attacker_1_Smooth'] = df['Agent_1_Reward'].rolling(window=window_size, min_periods=1).mean()
    df['Defender_2_Smooth'] = df['Agent_2_Reward'].rolling(window=window_size, min_periods=1).mean()

    print(f"Generating Task 3 academic plots in {save_dir}...")

    # --- PLOT 1: Overall Team Impact ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="Iteration", y="Mean_Reward", color="lightblue", alpha=0.3)
    sns.lineplot(data=df, x="Iteration", y="Mean_Reward_Smooth", color="navy", linewidth=2.5, label="Team Mean Reward")

    plt.title("Task 3: Overall Team Impact (Heterogeneous Roles)", fontweight='bold', pad=15)
    plt.xlabel("Training Iterations", fontweight='bold')
    plt.ylabel("Mean Episode Reward", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_roles_team_reward.png"), dpi=300)
    plt.close()

    # --- PLOT 2: Tactical Efficiency ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="Iteration", y="Episode_Length", color="thistle", alpha=0.3)
    sns.lineplot(data=df, x="Iteration", y="Episode_Length_Smooth", color="purple", linewidth=2.5,
                 label="Episode Length")

    plt.title("Task 3: Match Duration Dynamics", fontweight='bold', pad=15)
    plt.xlabel("Training Iterations", fontweight='bold')
    plt.ylabel("Average Steps per Match", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "02_roles_episode_length.png"), dpi=300)
    plt.close()

    # --- PLOT 3: THE MONEY SHOT - Role Specialization ---
    plt.figure(figsize=(8, 5))

    # Attackers plotted in warm colors (Reds/Oranges)
    sns.lineplot(data=df, x="Iteration", y="Attacker_0_Smooth", label="Agent 0 (Attacker)", color="crimson",
                 linewidth=2.5)
    sns.lineplot(data=df, x="Iteration", y="Attacker_1_Smooth", label="Agent 1 (Attacker)", color="darkorange",
                 linewidth=2.5, linestyle="--")

    # Defender plotted in a cool color (Blue) to show contrast
    sns.lineplot(data=df, x="Iteration", y="Defender_2_Smooth", label="Agent 2 (Defender)", color="royalblue",
                 linewidth=2.5)

    plt.title("Task 3: Evidence of Role Specialization", fontweight='bold', pad=15)
    plt.xlabel("Training Iterations", fontweight='bold')
    plt.ylabel("Smoothed Individual Reward", fontweight='bold')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_role_divergence.png"), dpi=300)
    plt.close()

    print(f"SUCCESS: Saved 3 high-resolution plots to {save_dir}/")


if __name__ == '__main__':
    # Hardcoded to the exact CSV output from your train_roles_ppo.py script
    csv_file = "ppo_v3_custom_roles_metrics.csv"
    generate_role_plots(csv_file)