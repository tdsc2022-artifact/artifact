from argparse import ArgumentParser
from os import path, remove, listdir, system
from typing import Dict, Tuple, List

import numpy
from tqdm import tqdm

from utils.filesystem import count_lines_in_file
from random import shuffle, seed
from omegaconf import DictConfig
from utils.unique import getMD5
from sklearn.model_selection import train_test_split
from typing import Counter as TypeCounter
from collections import Counter
from utils.vocabulary import Vocabulary_c2s, SOS, EOS, PAD, UNK
from utils.converting import parse_token


def _get_id2value_from_csv(path_: str) -> Dict[str, str]:
    return dict(numpy.genfromtxt(path_, delimiter=",", dtype=(str, str))[1:])


def preprocess_csv(data_folder: str, dataset_name: str, holdout_name: str,
                   is_shuffled: bool):
    """
    Preprocessing for files tokens.csv, paths.csv, node_types.csv
    """
    dataset_path = path.join(data_folder, dataset_name)
    id_to_token_data_path = path.join(dataset_path,
                                      f"tokens.{holdout_name}.csv")
    id_to_type_data_path = path.join(dataset_path,
                                     f"node_types.{holdout_name}.csv")
    id_to_paths_data_path = path.join(dataset_path,
                                      f"paths.{holdout_name}.csv")
    path_contexts_path = path.join(dataset_path,
                                   f"path_contexts.{holdout_name}.csv")
    output_c2s_path = path.join(dataset_path,
                                f"{dataset_name}.{holdout_name}.c2s")

    id_to_paths_stored = _get_id2value_from_csv(id_to_paths_data_path)
    id_to_paths = {
        index: [n for n in nodes.split()]
        for index, nodes in id_to_paths_stored.items()
    }

    id_to_node_types = _get_id2value_from_csv(id_to_type_data_path)
    id_to_node_types = {
        index: node_type.rsplit(" ", maxsplit=1)[0]
        for index, node_type in id_to_node_types.items()
    }

    id_to_tokens = _get_id2value_from_csv(id_to_token_data_path)

    if path.exists(output_c2s_path):
        remove(output_c2s_path)
    with open(path_contexts_path,
              "r") as path_contexts_file, open(output_c2s_path,
                                               "a+") as c2s_output:
        output_lines = []
        for line in tqdm(path_contexts_file,
                         total=count_lines_in_file(path_contexts_path)):
            label, *path_contexts = line.split()
            parsed_line = [label]
            for path_context in path_contexts:
                from_token_id, path_types_id, to_token_id = path_context.split(
                    ",")
                from_token, to_token = id_to_tokens[
                    from_token_id], id_to_tokens[to_token_id]
                nodes = [
                    id_to_node_types[p_] for p_ in id_to_paths[path_types_id]
                ]
                parsed_line.append(",".join(
                    [from_token, "|".join(nodes), to_token]))
            output_lines.append(" ".join(parsed_line + ["\n"]))
        if is_shuffled:
            shuffle(output_lines)
        c2s_output.write("".join(output_lines))


def astminer_to_c2s(config: DictConfig):

    astminer_res_root = path.join(config.data_folder, 'c2v')
    cweid = config.dataset.name
    dataset_root = path.join(astminer_res_root, cweid)
    vulTags = listdir(dataset_root)
    if (not path.exists(dataset_root) or len(vulTags) == 0):
        print("no csvs!")
        return
    out_root = path.join(config.data_folder, config.name, config.dataset.name)
    if not path.exists(out_root):
        system(f"mkdir -p {out_root}")
    output_c2s_path = path.join(out_root, "all.c2s")
    if (path.exists(output_c2s_path)):
        print("visited!")
        return
    output_train_c2s_path = path.join(out_root, "train.c2s")
    output_test_c2s_path = path.join(out_root, "test.c2s")
    output_val_c2s_path = path.join(out_root, "val.c2s")
    output_lines = []
    labels = []
    out_path_contexts = []
    out_path_contexts_list = []
    token_counter: TypeCounter[str] = Counter()
    node_counter: TypeCounter[str] = Counter()
    for vulTag in tqdm(vulTags, total=len(vulTags)):
        vulTag_root = path.join(dataset_root, vulTag)
        id_to_token_data_path = path.join(vulTag_root, "tokens.csv")
        id_to_type_data_path = path.join(vulTag_root, "node_types.csv")
        id_to_paths_data_path = path.join(vulTag_root, "paths.csv")
        path_contexts_path = path.join(vulTag_root, "path_contexts.csv")
        id_to_paths_stored = _get_id2value_from_csv(id_to_paths_data_path)
        id_to_paths = {
            index: [n for n in nodes.split()]
            for index, nodes in id_to_paths_stored.items()
        }

        id_to_node_types = _get_id2value_from_csv(id_to_type_data_path)
        id_to_node_types = {
            index: node_type.rsplit(" ", maxsplit=1)[0]
            for index, node_type in id_to_node_types.items()
        }

        id_to_tokens = _get_id2value_from_csv(id_to_token_data_path)

        if path.exists(output_c2s_path):
            remove(output_c2s_path)
        with open(path_contexts_path, "r") as path_contexts_file:
            for line in tqdm(path_contexts_file,
                             total=count_lines_in_file(path_contexts_path)):
                label, *path_contexts = line.split()
                labels.append(label)

                parsed_line = [label]
                parsed_context = []
                for path_context in path_contexts:
                    from_token_id, path_types_id, to_token_id = path_context.split(
                        ",")
                    from_token, to_token = id_to_tokens[
                        from_token_id], id_to_tokens[to_token_id]
                    nodes = [
                        id_to_node_types[p_]
                        for p_ in id_to_paths[path_types_id]
                    ]
                    parsed_line.append(",".join(
                        [from_token, "|".join(nodes), to_token]))
                    parsed_context.append(",".join(
                        [from_token, "|".join(nodes), to_token]))
                out_path_contexts.append(" ".join(parsed_context))
                out_path_contexts_list.append(parsed_context)
                output_lines.append(" ".join(parsed_line + ["\n"]))
    if (len(output_lines) == 0):
        print("no samples!")
        return
    # shuffle(output_lines)
    # unique outputlines {md5: [label, output line]}
    md5_to_lines: Dict[str, List[str, str]] = dict()
    mul_ct = 0
    conflict_ct = 0
    for idx, output_line in tqdm(enumerate(output_lines),
                                 total=len(output_lines)):
        md5 = getMD5(out_path_contexts[idx])
        label = labels[idx]
        if md5 not in md5_to_lines:
            md5_to_lines[md5] = [
                label, output_line, out_path_contexts_list[idx]
            ]
        else:
            md5_label = md5_to_lines[md5][0]
            if md5_label != -1 and md5_label != label:
                conflict_ct += 1
                md5_to_lines[md5][0] = -1
            else:
                mul_ct += 1

    print(f"total conflict: {conflict_ct}")
    print(f"total multiple: {mul_ct}")

    # build vocab
    output_lines = []
    for md5 in md5_to_lines:
        if md5_to_lines[md5][0] != -1:
            label = md5_to_lines[md5][0]
            line = md5_to_lines[md5][1]
            path_contexts = md5_to_lines[md5][2]
            output_lines.append(line)
            cur_tokens = []
            cur_nodes = []
            for path_context in path_contexts:

                from_token, path_nodes, to_token = path_context.split(",")
                cur_tokens += parse_token(from_token,
                                          config.dataset.token.is_splitted)
                cur_tokens += parse_token(to_token,
                                          config.dataset.token.is_splitted)
                cur_nodes += parse_token(path_nodes,
                                         config.dataset.path.is_splitted)
            token_counter.update(cur_tokens)
            node_counter.update(cur_nodes)

    with open(output_c2s_path, "a+") as c2s_output:
        c2s_output.write("".join(output_lines))

    vocab = _counters_to_vocab(config, token_counter, node_counter)
    vocab.dump_vocabulary(path.join(out_root, "vocab.pkl"))
    # split dataset
    print("start - split c2s...")

    X_train, X_test = train_test_split(output_lines, test_size=0.2)
    X_test, X_val = train_test_split(
        X_test,
        test_size=0.5,
    )
    print("done!")
    with open(output_train_c2s_path, "a+") as c2s_output:
        c2s_output.write("".join(X_train))
    with open(output_test_c2s_path, "a+") as c2s_output:
        c2s_output.write("".join(X_test))
    with open(output_val_c2s_path, "a+") as c2s_output:
        c2s_output.write("".join(X_val))


def _counter_to_dict(values: Counter,
                     n_most_common: int = None,
                     additional_values: List[str] = None) -> Dict[str, int]:
    dict_values = []
    if additional_values is not None:
        dict_values += additional_values
    dict_values += list(zip(*values.most_common(n_most_common)))[0]
    return {value: i for i, value in enumerate(dict_values)}


def _counters_to_vocab(config: DictConfig, token_counter: Counter,
                       node_counter: Counter) -> Vocabulary_c2s:
    names_additional_tokens = [
        SOS, EOS, PAD, UNK
    ] if config.dataset.token.is_wrapped else [PAD, UNK]
    token_to_id = _counter_to_dict(token_counter,
                                   config.dataset.token.vocabulary_size,
                                   names_additional_tokens)
    paths_additional_tokens = [
        SOS, EOS, PAD, UNK
    ] if config.dataset.path.is_wrapped else [PAD, UNK]
    node_to_id = _counter_to_dict(node_counter, None, paths_additional_tokens)

    vocabulary = Vocabulary_c2s(token_to_id=token_to_id, node_to_id=node_to_id)
    return vocabulary


def preprocess(config: DictConfig):
    astminer_to_c2s(config)
