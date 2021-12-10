from dataclasses import dataclass
from typing import List

import numpy
import torch


@dataclass
class SYSSample:
    tokens: numpy.ndarray
    label: int
    metric_info: numpy.ndarray
    n_tokens: int


class SYSBatch:
    def __init__(self, samples: List[SYSSample]):
        self.gadgets = torch.cat([
            torch.tensor(sample.tokens, dtype=torch.float32)
            for sample in samples
        ],
                                 dim=0)  # [total word; embedding size]

        self.labels = torch.from_numpy(
            numpy.array([sample.label for sample in samples]))
        self.metric_infos = torch.from_numpy(
            numpy.array([sample.metric_info for sample in samples]))
        self.tokens_per_label = torch.from_numpy(
            numpy.array([sample.n_tokens for sample in samples]))

    def __len__(self) -> int:
        return self.labels.size(0)

    def pin_memory(self) -> "SYSBatch":
        self.gadgets = self.gadgets.pin_memory()
        self.labels = self.labels.pin_memory()
        self.metric_infos = self.metric_infos.pin_memory()
        self.tokens_per_label = self.tokens_per_label.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        # self.metric_infos = self.metric_infos.to(device)
        self.tokens_per_label = self.tokens_per_label.to(device)
        self.gadgets = self.gadgets.to(device)
