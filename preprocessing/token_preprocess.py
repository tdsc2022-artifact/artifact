from omegaconf import DictConfig
from utils.vocabulary import Vocabulary_token
from collections import Counter
from typing import Counter as TypeCounter
from os.path import join
from tqdm import tqdm
from utils.filesystem import count_lines_in_file
from utils.vocabulary import PAD, UNK
from typing import List, Dict
from utils.common import get_config
from utils.unique import unique_token
import os
from sklearn.model_selection import train_test_split


def _counter_to_dict(values: Counter,
                     n_most_common: int = None,
                     additional_values: List[str] = None) -> Dict[str, int]:
    dict_values = []
    if additional_values is not None:
        dict_values += additional_values
    dict_values += list(zip(*values.most_common(n_most_common)))[0]
    return {value: i for i, value in enumerate(dict_values)}


def collect_vocab(config: DictConfig) -> Vocabulary_token:
    cweid = config.dataset.name
    token_counter: TypeCounter[str] = Counter()
    data_path = join(config.data_folder, "token", cweid, f"{cweid}.txt")
    with open(data_path, "r") as f:
        for line in tqdm(f, total=count_lines_in_file(data_path)):
            label, *tokens = line.split()
            token_counter.update(tokens)
    additional_tokens = [PAD, UNK]
    token_to_id = _counter_to_dict(token_counter,
                                   config.dataset.token.vocabulary_size,
                                   additional_tokens)
    vocab = Vocabulary_token(token_to_id)
    vocab.dump_vocabulary(
        join(config.data_folder, "token", cweid, f"vocab.pkl"))


def preprocess(config: DictConfig):
    # if (os.path.exists(
    #         join(config.data_folder, "token", config.dataset.name,
    #              f"vocab.pkl"))):
    #     return
    # unique tokens
    cweid = config.dataset.name
    data_path = join(config.data_folder, "token", cweid, "all.txt")
    if (not os.path.exists(
            join(config.data_folder, 'token', cweid, f'{cweid}_mul.txt'))):
        os.system(
            f"cp {data_path} {join(config.data_folder, 'token', cweid, f'{cweid}_mul.txt')}"
        )
    tokens = list()
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            target, *content = line.split()
            token = dict()
            token["target"] = target
            token["content"] = content
            tokens.append(token)
    md5Dict = unique_token(tokens)
    res = ""
    for md5 in tqdm(md5Dict, total=len(md5Dict)):
        target = md5Dict[md5]["target"]
        if target != -1:
            res = res + target + " "
            res += " ".join(md5Dict[md5]["content"])
            res += "\n"
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(res)
    collect_vocab(config)
    split_dataset(config)


def split_dataset(config: DictConfig):
    # unique tokens
    cweid = config.dataset.name
    data_path = join(config.data_folder, "token", cweid, f"{cweid}.txt")
    train_data_path = join(config.data_folder, "token", cweid,
                           "train.txt")
    test_data_path = join(config.data_folder, "token", cweid,
                          "test.txt")
    val_data_path = join(config.data_folder, "token", cweid,
                         "val.txt")
    if (os.path.exists(train_data_path)):
        os.system(f"rm {train_data_path}")
    if (os.path.exists(test_data_path)):
        os.system(f"rm {test_data_path}")
    if (os.path.exists(val_data_path)):
        os.system(f"rm {val_data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    X_train, X_test = train_test_split(lines, test_size=0.2)
    X_test, X_val = train_test_split(
        X_test,
        test_size=0.5,
    )
    with open(train_data_path, "a") as f:
        for line in X_train:
            f.write(line)
    with open(test_data_path, "a") as f:
        for line in X_test:
            f.write(line)
    with open(val_data_path, "a") as f:
        for line in X_val:
            f.write(line)


if __name__ == "__main__":
    _config = get_config("token", "example")
    collect_vocab(_config)
