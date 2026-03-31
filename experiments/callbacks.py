import json
import os
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class StepRewardCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        config = getattr(worker, "config", {})
        if isinstance(config, dict):
            run_name = config.get("env_config", {}).get("run_name", "unknown_run")
        else:
            run_name = getattr(config, "env_config", {}).get("run_name", "unknown_run")

        # FIX: Ensure agent IDs are strings and rewards are floats
        final_scores = {str(aid): float(rew) for aid, rew in episode.agent_rewards.items()}

        data_entry = {
            "episode_id": str(episode.episode_id), # FIX: Convert tuple ID to string
            "episode_length": int(episode.length),
            "final_rewards": final_scores
        }

        save_dir = os.path.join("train_data", run_name)
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, "episode_rewards.json")

        with open(json_path, 'a') as f:
            f.write(json.dumps(data_entry) + "\n")