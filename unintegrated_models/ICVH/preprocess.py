#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Author      :ives-nx
@version      :1.0
'''

from collections import defaultdict
from email.policy import default
import sys
from turtle import st

from regex import F
sys.path.append("../..")
from typing import List
from utils.vectorize_gadget import GadgetVectorizer
from utils.clean_gadget import clean_gadget
import json

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


def symbolic(functions:List) -> List:
    """
    @description  :
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    for func in functions:
        code = func['code']
        clean_code = clean_gadget(code)
        func['code_sym'] = clean_code
    return functions

def count_stmts_frequency(functions:List):
    freq_dict = defaultdict(int)
    stmts_count = 0
    for sample in functions:
        for stmt in sample["code_sym"]:
            freq_dict[stmt] += 1
        stmts_count+=len(sample["code_sym"])
    for stmt in freq_dict:
        freq = freq_dict[stmt] / stmts_count
        freq_dict[stmt] = freq
    return freq_dict




def _counter_to_dict(values: Counter,
                     n_most_common: int = None,
                     additional_values: List[str] = None) -> Dict[str, int]:
    dict_values = []
    if additional_values is not None:
        dict_values += additional_values
    dict_values += list(zip(*values.most_common(n_most_common)))[0]
    return {value: i for i, value in enumerate(dict_values)}


def collect_vocab(functions:List) -> Vocabulary_token:
    token_counter: TypeCounter[str] = Counter()
    
    for sample in tqdm(functions):
        tokens = sample['code_sym']
        token_counter.update(tokens)
    additional_tokens = [PAD, UNK]
    token_to_id = _counter_to_dict(token_counter,
                                   20000,
                                   additional_tokens)
    vocab = Vocabulary_token(token_to_id)
    vocab.dump_vocabulary(
        join(f"data/vocab.pkl"))





    