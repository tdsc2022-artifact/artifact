from dataclasses import dataclass
from typing import List

import numpy
import torch


@dataclass
class VDPSample:
    tokens: numpy.ndarray
    label: int
    is_back: bool
    n_tokens: int


class VDPBatch:
    def __init__(self, samples: List[VDPSample]):
        self.gadgets = torch.cat([
            torch.tensor(sample.tokens, dtype=torch.float32)
            for sample in samples
        ],
                                 dim=0)  # [total word; embedding size]

        self.labels = torch.from_numpy(
            numpy.array([sample.label for sample in samples]))
        self.is_back = torch.from_numpy(
            numpy.array([sample.is_back for sample in samples]))
        self.tokens_per_label = torch.from_numpy(
            numpy.array([sample.n_tokens for sample in samples]))

    def __len__(self) -> int:
        return self.labels.size(0)

    def pin_memory(self) -> "VDPBatch":
        self.gadgets = self.gadgets.pin_memory()
        self.labels = self.labels.pin_memory()
        self.is_back = self.is_back.pin_memory()
        self.tokens_per_label = self.tokens_per_label.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.tokens_per_label = self.tokens_per_label.to(device)
        self.gadgets = self.gadgets.to(device)
        self.is_back = self.is_back.to(device)
