from dataclasses import dataclass
from typing import List

import numpy
import torch


@dataclass
class MulVDPSample:
    global_tokens: numpy.ndarray
    global_n_tokens: int
    local_tokens: numpy.ndarray
    local_n_tokens: int
    label: int


class MulVDPBatch:
    def __init__(self, samples: List[MulVDPSample]):
        self.gadgets = torch.cat([
            torch.tensor(sample.global_tokens, dtype=torch.float32)
            for sample in samples
        ],
                                 dim=0)  # [total word; embedding size]

        self.attns = torch.cat([
            torch.tensor(sample.local_tokens, dtype=torch.float32)
            for sample in samples
        ],
                                 dim=0)  # [total word; embedding size]

        self.labels = torch.from_numpy(
            numpy.array([sample.label for sample in samples]))
        self.global_tokens_per_label = torch.from_numpy(
            numpy.array([sample.global_n_tokens for sample in samples]))
        self.local_tokens_per_label = torch.from_numpy(
            numpy.array([sample.local_n_tokens for sample in samples]))

    def __len__(self) -> int:
        return self.labels.size(0)

    def pin_memory(self) -> "MulVDPBatch":
        self.gadgets = self.gadgets.pin_memory()
        self.attns = self.attns.pin_memory()
        self.labels = self.labels.pin_memory()
        self.global_tokens_per_label = self.global_tokens_per_label.pin_memory()
        self.local_tokens_per_label = self.local_tokens_per_label.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.global_tokens_per_label = self.global_tokens_per_label.to(device)
        self.local_tokens_per_label = self.local_tokens_per_label.to(device)
        self.gadgets = self.gadgets.to(device)
        self.attns = self.attns.to(device)
