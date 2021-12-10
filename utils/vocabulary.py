import pickle
from dataclasses import dataclass
from os.path import exists
from typing import Dict, Optional, Counter, List

# vocabulary keys
TOKEN_TO_ID = "token_to_id"
NODE_TO_ID = "node_to_id"
LABEL_TO_ID = "label_to_id"

# sequence service tokens
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"


@dataclass
class Vocabulary_token:
    token_to_id: Dict[str, int]

    def add_from_counter(
        self,
        target_field: str,
        counter: Counter,
        n_most_values: int = -1,
        add_values: List[str] = None,
    ):
        if not hasattr(self, target_field):
            raise ValueError(
                f"There is no {target_field} attribute in vocabulary class")
        if add_values is None:
            add_values = []
        if n_most_values == -1:
            add_values += list(zip(*counter.most_common()))[0]
        else:
            add_values += list(
                zip(*counter.most_common(n_most_values - len(add_values))))[0]
        attr = {value: i for i, value in enumerate(add_values)}
        setattr(self, target_field, attr)

    @staticmethod
    def load_vocabulary(vocabulary_path: str) -> "Vocabulary_token":
        if not exists(vocabulary_path):
            raise ValueError(f"Can't find vocabulary in: {vocabulary_path}")
        with open(vocabulary_path, "rb") as vocabulary_file:
            vocabulary_dicts = pickle.load(vocabulary_file)
        token_to_id = vocabulary_dicts[TOKEN_TO_ID]
        return Vocabulary_token(token_to_id=token_to_id)

    def dump_vocabulary(self, vocabulary_path: str):
        with open(vocabulary_path, "wb") as vocabulary_file:
            vocabulary_dicts = {
                TOKEN_TO_ID: self.token_to_id,
            }
            pickle.dump(vocabulary_dicts, vocabulary_file)


@dataclass
class Vocabulary_c2s:
    token_to_id: Dict[str, int]
    node_to_id: Dict[str, int]

    @staticmethod
    def load_vocabulary(vocabulary_path: str) -> "Vocabulary_c2s":
        if not exists(vocabulary_path):
            raise ValueError(f"Can't find vocabulary in: {vocabulary_path}")
        with open(vocabulary_path, "rb") as vocabulary_file:
            vocabulary_dicts = pickle.load(vocabulary_file)
        token_to_id = vocabulary_dicts[TOKEN_TO_ID]
        node_to_id = vocabulary_dicts[NODE_TO_ID]
        return Vocabulary_c2s(token_to_id=token_to_id,
                              node_to_id=node_to_id)

    def dump_vocabulary(self, vocabulary_path: str):
        with open(vocabulary_path, "wb") as vocabulary_file:
            vocabulary_dicts = {
                TOKEN_TO_ID: self.token_to_id,
                NODE_TO_ID: self.node_to_id
            }
            pickle.dump(vocabulary_dicts, vocabulary_file)
