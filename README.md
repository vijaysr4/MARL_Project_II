# APM_5DA01_TP: MARL Project

This repository contains the training and evaluation pipeline for a Multi-Agent Reinforcement Learning (MARL) team playing a 3v3 maritime Capture the Flag game using the [PyQuaticus](https://github.com/mit-ll/pyquaticus) environment and Ray RLlib.

The project explores both **Independent PPO (IPPO)** and **Multi-Agent PPO (MAPPO)** with a Centralized Critic, coupled with custom Reward Shaping to assign specific offensive and defensive roles.

### Demo: MAPPO (50 Iterations) vs. MIT Heuristic Bots

![MAPPO vs Heuristic Match](link-to-your-gif-here.gif)

> **Note on Training Depth:** The visual above demonstrates our MAPPO agents (with Shaped Rewards) surviving the maximum 600 steps against the hardcoded, top-tier MIT Heuristic bots. It is important to note that 50 iterations (roughly 50 x 2,000 steps against random agents) is effectively the early learning phase for a reinforcement learning agent. While they have not yet mastered complex team maneuvers to capture the flag, they successfully learned to avoid boundaries (Out of Bounds penalties) and survive a full match without being captured by a vastly superior enemy.

---

## Repository Structure

```text
.
├── experiments/                 # Core Pipeline
│   ├── train_unified.py         # Main training script (supports paradigm switching and phase shifts)
│   ├── eval_model.py            # Evaluation script (generates JSON logs and matplotlib charts)
│   └── callbacks.py             # Custom RLlib callbacks to securely log step-by-step match data
├── pyquaticus/                  # Custom Modules
│   ├── models/
│   │   └── marl_models.py       # Custom CentralizedCriticModel handling dictionary observation spaces
│   └── utils/
│       └── rewards.py           # Custom reward shaping functions (attacker and defender specific)
├── rl_test/                     # Archive containing legacy tests and scripts from earlier phases
├── trained_models/              # Auto-generated: Stores PyTorch training checkpoints
└── results_eval/                # Auto-generated: Stores JSON match logs and performance graphs
````

-----

## Architecture Highlights

### 1\. MAPPO: Centralized Critic & Decentralized Actor

To enable true team coordination, we implemented a custom model in `marl_models.py`. By utilizing a `GlobalObservationWrapper`, our environment passes a comprehensive dictionary containing both local and global data.

  * **The Actor Network** dynamically unpacks the dictionary, taking only its local 61-dimension observation to make decentralized decisions during execution.
  * **The Critic Network** utilizes the full 366-dimension global state to accurately evaluate the value function, enabling the agents to learn from the entire field's state during centralized training.

### 2\. Reward Shaping & Role Assignment

Instead of relying purely on sparse capture rewards, we built custom reward functions in `rewards.py` to shape specific team behaviors:

  * **Attackers (Agents 0 & 1):** Rewarded for moving closer to the enemy flag and heavily penalized for hitting boundaries.
  * **Defender (Agent 2):** Given a strict distance penalty if it wanders more than 20 meters from the home base, forcing it to guard the flag perimeter.

-----

## Usage

### Training

The unified training script allows you to easily swap between IPPO and MAPPO, as well as Sparse or Shaped rewards. The script includes a **Phase Shift**: at iteration 26, the Red Team automatically upgrades from Random bots to MIT Heuristic bots to increase the difficulty mid-training.

```bash
# Train the MAPPO model with Custom Roles
python experiments/train_unified.py --paradigm MAPPO --reward SHAPED --iters 50

# Train the Baseline IPPO model
python experiments/train_unified.py --paradigm IPPO --reward SPARSE --iters 50
```

### Evaluation

The evaluation script loads your trained checkpoints and runs a 1,000-step simulation against your opponent of choice, outputting match logs and performance graphs.

```bash
# Evaluate against the Hard Mode MIT bots
python experiments/eval_model.py --paradigm MAPPO --reward SHAPED --iter 50 --opponent heuristic

# Evaluate against Baseline Random bots
python experiments/eval_model.py --paradigm IPPO --reward SPARSE --iter 50 --opponent random
```

-----

### Acknowledgements

Built using [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) and the [MIT Lincoln Laboratory PyQuaticus Environment](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/mit-ll/pyquaticus).

