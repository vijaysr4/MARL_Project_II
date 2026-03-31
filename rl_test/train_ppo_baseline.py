import os
import csv
from ray.rllib.algorithms.ppo import PPOConfig
from pyquaticus import pyquaticus_v0
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus.config import config_dict_std
from ray.tune.registry import register_env

if __name__ == '__main__':
    # No custom rewards = Sparse Baseline
    reward_config = {'agent_0': None, 'agent_1': None, 'agent_2': None}

    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    env_creator = lambda _: pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict, reward_config=reward_config,
                                                        team_size=3)
    register_env('pyquaticus_sparse', lambda config: ParallelPettingZooWrapper(env_creator(config)))

    ppo_config = (PPOConfig().environment(env='pyquaticus_sparse').env_runners(num_env_runners=1))
    algo = ppo_config.build()

    csv_filename = "ppo_baseline_metrics.csv"
    with open(csv_filename, mode='w', newline='') as file:
        csv.writer(file).writerow(["Iteration", "Mean_Reward"])

    print("Generating PPO Sparse Baseline Data...")
    for i in range(50):
        result = algo.train()
        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', 0)
        with open(csv_filename, mode='a', newline='') as file:
            csv.writer(file).writerow([i, mean_reward])
        print(f"Iter {i}: {mean_reward}")