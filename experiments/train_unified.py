import os
import time
import argparse
import numpy as np
import ray
import logging
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.tune.logger import UnifiedLogger

from pyquaticus import pyquaticus_v0
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus.base_policies.base_policy_wrappers import DefendGen, AttackGen
from pyquaticus.envs.observation_wrapper import GlobalObservationWrapper
from pyquaticus.models.marl_models import CentralizedCriticModel
import pyquaticus.utils.rewards as rew
from callbacks import StepRewardCallbacks


# --- Heuristic Wrappers FIX: Added team_size=3 ---
class RLlibHeuristicAttack(Policy):
    def __init__(self, obs_space, act_space, config):
        Policy.__init__(self, obs_space, act_space, config)
        from pyquaticus.base_policies.base_policy_wrappers import AttackGen
        self.internal_policy = AttackGen('agent_5', pyquaticus_v0.PyQuaticusEnv(team_size=3), 'easy')(obs_space,
                                                                                                      act_space, config)

    def compute_actions(self, obs_batch, *args, **kwargs):
        return self.internal_policy.compute_actions(obs_batch, *args, **kwargs)

    # Added to satisfy RLlib Policy requirements
    def get_weights(self): return {}

    def set_weights(self, weights): pass


class RLlibHeuristicDefend(Policy):
    def __init__(self, obs_space, act_space, config):
        Policy.__init__(self, obs_space, act_space, config)
        from pyquaticus.base_policies.base_policy_wrappers import DefendGen
        self.internal_policy = DefendGen('agent_3', pyquaticus_v0.PyQuaticusEnv(team_size=3), 'easy')(obs_space,
                                                                                                      act_space, config)

    def compute_actions(self, obs_batch, *args, **kwargs):
        return self.internal_policy.compute_actions(obs_batch, *args, **kwargs)

    # Added to satisfy RLlib Policy requirements
    def get_weights(self): return {}

    def set_weights(self, weights): pass

OPPONENT_PHASE = "random"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paradigm", type=str, choices=["IPPO", "MAPPO"], default="IPPO")
    parser.add_argument("--reward", type=str, choices=["SPARSE", "SHAPED"], default="SPARSE")
    parser.add_argument("--iters", type=int, default=50)
    # --- ADD THIS LINE FOR TESTING ---
    parser.add_argument("--switch-iter", type=int, default=25, help="Iter to switch opponents")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
    RUN_NAME = f"{args.paradigm}_{args.reward}"

    if args.reward == "SHAPED":
        reward_config = {'agent_0': rew.pure_attacker, 'agent_1': rew.pure_attacker, 'agent_2': rew.pure_defender}
    else:
        reward_config = {f'agent_{i}': rew.caps_and_grabs for i in range(3)}
    reward_config.update({f'agent_{i}': None for i in range(3, 6)})

    def env_creator(config):
        base_env = pyquaticus_v0.PyQuaticusEnv(reward_config=reward_config, team_size=3)
        if args.paradigm == "MAPPO":
            base_env = GlobalObservationWrapper(base_env)
        return ParallelPettingZooWrapper(base_env)

    register_env('pyquaticus', env_creator)
    test_env = env_creator({})
    obs_space = test_env.observation_space['agent_0']
    act_space = test_env.action_space['agent_0']
    test_env.close()

    # Define Policies
    if args.paradigm == "IPPO":
        policies = {f"pol_{i}": (None, obs_space, act_space, {}) for i in range(3)}
    else:
        policies = {"shared_pol": (None, obs_space, act_space, {"model": {"custom_model": "cc_model"}})}

    policies.update({
        "random_opp": (None, obs_space, act_space, {}),
        "heuristic_def": (RLlibHeuristicDefend, obs_space, act_space, {}),
        "heuristic_att": (RLlibHeuristicAttack, obs_space, act_space, {})
    })


    # Changed *args to episode, worker to avoid shadowing your global 'args'
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # Determine if it is a MAPPO run by checking the policies keys
        is_mappo = "shared_pol" in policies

        # Determine Blue Team (Agents 0, 1, 2)
        if int(agent_id[-1]) < 3:
            return "shared_pol" if is_mappo else f"pol_{agent_id[-1]}"

        # Determine Red Team (Agents 3, 4, 5)
        if OPPONENT_PHASE == "random":
            return "random_opp"
        else:
            # 3 and 4 Defend, 5 Attacks
            return "heuristic_def" if agent_id in ['agent_3', 'agent_4'] else "heuristic_att"

    config = (PPOConfig()
              .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
              .environment(env='pyquaticus', env_config={"run_name": RUN_NAME})
              .env_runners(num_env_runners=1)
              .callbacks(StepRewardCallbacks)
              .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn,
                               policies_to_train=[p for p in policies.keys() if "heuristic" not in p and "opp" not in p]))

    algo = config.build_algo(logger_creator=lambda cfg: UnifiedLogger(cfg, f"./ray_results/{RUN_NAME}", loggers=None))

    for i in range(args.iters + 1):
        if i == 26:  # Or args.switch_iter if you added the argument
            print("\n--- PHASE SHIFT: Opponents are now HEURISTIC ---")


            def set_phase_heuristic(env_runner):
                import __main__
                __main__.OPPONENT_PHASE = "heuristic"


            # THIS is the line that fixes the crash:
            algo.env_runner_group.foreach_env_runner(set_phase_heuristic)

            OPPONENT_PHASE = "heuristic"

        result = algo.train()
        print(f"Iter: {i} | Mean Reward: {result.get('env_runners', {}).get('episode_reward_mean', 0):.2f}")

        if i % 25 == 0 or i == args.iters:
            algo.save(os.path.abspath(os.path.join("trained_models", RUN_NAME, f"iter_{i}")))

    ray.shutdown()