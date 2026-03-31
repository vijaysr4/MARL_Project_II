import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import collections


class CentralizedCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. Determine space dimensions
        # FIX: Check for the 'spaces' attribute used by Gymnasium instead of python 'dict'
        original_space = getattr(obs_space, "original_space", None)

        if original_space is not None and hasattr(original_space, "spaces") and "global_state" in original_space.spaces:
            self.is_dict = True
            self.actor_input_dim = original_space.spaces["obs"].shape[0]  # 61
            self.critic_input_dim = original_space.spaces["global_state"].shape[0]  # 366
        else:
            self.is_dict = False
            self.actor_input_dim = obs_space.shape[0]
            self.critic_input_dim = obs_space.shape[0]

        # 2. THE ACTOR (Standard PyTorch Sequential)
        self.actor_net = nn.Sequential(
            nn.Linear(self.actor_input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

        # 3. THE CRITIC (Standard PyTorch Sequential)
        self.critic_net = nn.Sequential(
            nn.Linear(self.critic_input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self._current_global_state = None

    def forward(self, input_dict, state, seq_lens):
        raw_obs = input_dict["obs"]

        if self.is_dict:
            # MAPPO Path
            if isinstance(raw_obs, dict) or isinstance(raw_obs, collections.OrderedDict):
                local_obs = raw_obs["obs"]
                self._current_global_state = raw_obs["global_state"]
            else:
                local_obs = raw_obs[:, :self.actor_input_dim]
                self._current_global_state = raw_obs[:, self.actor_input_dim:]
        else:
            # IPPO Path
            local_obs = raw_obs
            self._current_global_state = raw_obs

        # Safety Net
        if isinstance(local_obs, dict) or isinstance(local_obs, collections.OrderedDict):
            local_obs = local_obs.get("obs", local_obs)

        if isinstance(self._current_global_state, dict) or isinstance(self._current_global_state,
                                                                      collections.OrderedDict):
            self._current_global_state = self._current_global_state.get("global_state", self._current_global_state)

        if not isinstance(local_obs, torch.Tensor):
            local_obs = torch.tensor(local_obs)
        local_obs = local_obs.float()

        if not isinstance(self._current_global_state, torch.Tensor):
            self._current_global_state = torch.tensor(self._current_global_state)
        self._current_global_state = self._current_global_state.float()

        logits = self.actor_net(local_obs)
        return logits, state

    def value_function(self):
        return torch.reshape(self.critic_net(self._current_global_state), [-1])