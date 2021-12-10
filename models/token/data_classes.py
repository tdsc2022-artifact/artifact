from dataclasses import dataclass
from typing import List

import numpy
import torch


@dataclass
class TokensSample:
    tokens: numpy.ndarray
    label: int
    n_tokens: int


class TokensBatch:
    def __init__(self, samples: List[TokensSample]):
        self.labels = torch.from_numpy(numpy.array([_s.label for _s in samples]))  # [batch]
        self.tokens_per_label = torch.from_numpy(numpy.array([_s.n_tokens for _s in samples]))

        tokens = numpy.hstack([_s.tokens
                               for _s in samples])  # [seq len; batch]
        self.tokens = torch.from_numpy(tokens)

    def __len__(self) -> int:
        return self.labels.size(0)

    def pin_memory(self) -> "TokensBatch":
        self.tokens = self.tokens.pin_memory()
        self.labels = self.labels.pin_memory()
        self.tokens_per_label = self.tokens_per_label.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.tokens_per_label = self.tokens_per_label.to(device)
        self.tokens = self.tokens.to(device)
