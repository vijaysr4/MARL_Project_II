import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# --- Configure Matplotlib for a scientific/LaTeX aesthetic ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "grid.alpha": 0.3,
    "lines.linewidth": 2
})


def plot_marl_comparison(results_dir="./ray_results"):
    plt.figure(figsize=(10, 6))

    # Standard academic color palette
    colors = {
        "IPPO_SPARSE": "#d62728",  # Red
        "IPPO_SHAPED": "#1f77b4",  # Blue
        "MAPPO_SPARSE": "#ff7f0e",  # Orange
        "MAPPO_SHAPED": "#2ca02c"  # Green
    }

    # Recursive search for the high-resolution episode logs
    csv_files = glob.glob(os.path.join(results_dir, "**/episode_history.csv"), recursive=True)

    if not csv_files:
        print("No episode logs found in ray_results directory.")
        return

    for csv_path in csv_files:
        # Extract the run name from the directory structure
        path_parts = csv_path.split(os.sep)
        # Assumes structure: ray_results/RUN_NAME/PPO_.../episode_history.csv
        label = path_parts[-3]

        df = pd.read_csv(csv_path)

        # Calculate moving average to highlight trends over noise
        # 100 episodes is usually a good window for 1000 iterations
        df['smoothed_reward'] = df['episode_reward'].rolling(window=100, min_periods=10).mean()

        plt.plot(
            df['episode_id'],
            df['smoothed_reward'],
            label=label.replace("_", " "),
            color=colors.get(label)
        )

    plt.title("Multi-Agent Learning Performance: IPPO vs. MAPPO")
    plt.xlabel("Episode Number")
    plt.ylabel("Mean Episode Reward (Smoothed)")
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle='--')

    # Save as high-resolution image for presentations or papers
    output_name = "marl_training_comparison.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved as: {output_name}")


if __name__ == "__main__":
    plot_marl_comparison()