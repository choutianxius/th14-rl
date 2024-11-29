import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs
from typing import Optional, List, Type


class DuelingQNetwork(QNetwork):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super(DuelingQNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images,
        )
        del self.q_net

        if len(self.net_arch) == 0:
            raise ValueError("There should be at least 1 fully connected layer")

        action_dim = int(self.action_space.n)
        last_layer_dim = int(self.net_arch[-1])

        shared_layers = create_mlp(
            self.features_dim, last_layer_dim, self.net_arch[:-1], self.activation_fn
        )
        self.shared_layers = nn.Sequential(*shared_layers)

        # value stream
        self.V = nn.Sequential(nn.Linear(last_layer_dim, 1))
        # advantage stream
        self.A = nn.Sequential(nn.Linear(last_layer_dim, action_dim))

    def forward(self, obs: PyTorchObs) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        shared_out = self.shared_layers(features)
        values = self.V(shared_out)
        advantages = self.A(shared_out)
        q = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q


class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return DuelingQNetwork(**net_args).to(self.device)
