import subprocess
import os
import sys

paradigms = ["IPPO", "MAPPO"]
rewards = ["SPARSE", "SHAPED"]
iterations = 50  # Changed to 50 as requested

for p in paradigms:
    for r in rewards:
        run_name = f"{p}_{r}"
        print(f"--- Starting Run: {run_name} ---")
        cmd = [
            sys.executable, "experiments/train_unified.py",
            "--paradigm", p,
            "--reward", r,
            "--iters", str(iterations)
        ]
        subprocess.run(cmd, check=True)

print("All experiments complete.")

# python experiments/train_unified.py --paradigm IPPO --reward SPARSE --iters 50
#
# python experiments/train_unified.py --paradigm IPPO --reward SHAPED --iters 50
#
# python experiments/train_unified.py --paradigm MAPPO --reward SPARSE --iters 50
#
# python experiments/train_unified.py --paradigm MAPPO --reward SHAPED --iters 50
