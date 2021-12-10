import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy


@dataclass
class BufferedPathContext:
    """Class for storing buffered path contexts.

    :attr vectors: [total word len, embedding size]
    :attr labels: list [buffer size]
    :attr words_per_label: list [buffer size] -- number of words for each label

    """

    vectors: numpy.ndarray
    labels: List[int]
    is_back: List[bool]
    words_per_label: List[int]

    def __post_init__(self):
        self._end_idx = numpy.cumsum(self.words_per_label).tolist()
        self._start_idx = [0] + self._end_idx[:-1]

    def __len__(self):
        return len(self.words_per_label)

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, int, bool, int]:
        path_slice = slice(self._start_idx[idx], self._end_idx[idx])
        vector = self.vectors[path_slice, :]
        return vector, self.labels[idx], self.is_back[
            idx], self.words_per_label[idx]

    def dump(self, path: str):
        with open(path, "wb") as pickle_file:
            pickle.dump((self.vectors, self.labels, self.is_back,
                         self.words_per_label), pickle_file)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        if not isinstance(data, tuple) and len(data) != 4:
            raise RuntimeError("Incorrect data inside pickled file")
        return BufferedPathContext(*data)

    @staticmethod
    def create_from_lists(vectors: List[numpy.ndarray], labels: List[int],
                          is_back: List[bool]) -> "BufferedPathContext":
        merge_vectors = numpy.concatenate(vectors, axis=0)
        words_per_label = [len(vector) for vector in vectors]

        return BufferedPathContext(merge_vectors, labels, is_back,
                                   words_per_label)
