import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy


@dataclass
class BufferedPathContext:
    """Class for storing buffered path contexts.

    :attr global_vectors: (global total word len, embedding size)
    :attr local_vectors: (local total word len, embedding size)
    :attr global_words_per_label: list (buffer size) -- number of global words for each label
    :attr local_words_per_label: list (buffer size) -- number of local words for each label
    :attr labels: list (buffer size)
    """

    global_vectors: numpy.ndarray
    global_words_per_label: List[int]
    local_vectors: numpy.ndarray
    local_words_per_label: List[int]
    labels: List[int]

    def __post_init__(self):
        self._global_end_idx = numpy.cumsum(
            self.global_words_per_label).tolist()
        self._global_start_idx = [0] + self._global_end_idx[:-1]
        self._local_end_idx = numpy.cumsum(self.local_words_per_label).tolist()
        self._local_start_idx = [0] + self._local_end_idx[:-1]

    def __len__(self):
        return len(self.global_words_per_label)

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, int, bool, int]:
        global_path_slice = slice(self._global_start_idx[idx],
                                  self._global_end_idx[idx])
        global_vector = self.global_vectors[global_path_slice, :]
        local_path_slice = slice(self._local_start_idx[idx],
                                 self._local_end_idx[idx])
        local_vector = self.local_vectors[local_path_slice, :]
        return global_vector, local_vector, self.global_words_per_label[
            idx], self.local_words_per_label[idx], self.labels[idx]

    def dump(self, path: str):
        with open(path, "wb") as pickle_file:
            pickle.dump(
                (self.global_vectors, self.global_words_per_label,
                 self.local_vectors, self.local_words_per_label, self.labels),
                pickle_file)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        if not isinstance(data, tuple) and len(data) != 4:
            raise RuntimeError("Incorrect data inside pickled file")
        return BufferedPathContext(*data)

    @staticmethod
    def create_from_lists(global_vectors: List[numpy.ndarray],
                          local_vectors: List[numpy.ndarray],
                          labels: List[int]) -> "BufferedPathContext":
        global_merge_vectors = numpy.concatenate(global_vectors, axis=0)
        global_words_per_label = [len(vector) for vector in global_vectors]

        local_merge_vectors = numpy.concatenate(local_vectors, axis=0)
        local_words_per_label = [len(vector) for vector in local_vectors]

        return BufferedPathContext(global_merge_vectors,
                                   global_words_per_label, local_merge_vectors,
                                   local_words_per_label, labels)
