from os.path import exists

from omegaconf import DictConfig
from torch.utils.data import Dataset

from models.token.data_classes import TokensSample
from utils.converting import tokens_to_wrapped_numpy
from utils.vocabulary import Vocabulary_token


class TokenDataset(Dataset):

    _separator = " "

    def __init__(self, data_file_path: str, config: DictConfig,
                 vocabulary: Vocabulary_token):
        if not exists(data_file_path):
            raise ValueError(f"Can't find file with data: {data_file_path}")
        self._data_file_path = data_file_path
        self._hyper_parameters = config.hyper_parameters
        self._line_offsets = []
        cumulative_offset = 0
        with open(self._data_file_path, "r") as data_file:
            for line in data_file:
                self._line_offsets.append(cumulative_offset)
                cumulative_offset += len(line.encode(data_file.encoding))
        self._n_samples = len(self._line_offsets)

        self._token_vocabulary = vocabulary.token_to_id
        self._token_parameters = config.dataset.token

    def __len__(self):
        return self._n_samples

    def _read_line(self, index: int) -> str:
        with open(self._data_file_path, "r") as data_file:
            data_file.seek(self._line_offsets[index])
            line = data_file.readline().strip()
        return line

    def __getitem__(self, index) -> TokensSample:
        raw_sample = self._read_line(index)
        label, *tokens = raw_sample.split()

        # convert tokens to wrapped numpy array
        wrapped_tokens = tokens_to_wrapped_numpy(
            tokens,
            self._token_vocabulary,
            self._token_parameters.max_parts,
            self._token_parameters.is_wrapped,
        )
        n_tokens = min(self._token_parameters.max_parts, len(tokens)) + (
            1 if self._token_parameters.is_wrapped else 0)

        return TokensSample(tokens=wrapped_tokens,
                            label=int(label),
                            n_tokens=n_tokens)

    def get_n_samples(self):
        return self._n_samples