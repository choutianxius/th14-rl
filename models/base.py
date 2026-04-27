from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ModelBase(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor: ...

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(path)
