import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GridCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # 3

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size dynamically so it works for any grid size
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            flat_size = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
