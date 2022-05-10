import json
from lib2to3.pgen2.literals import simple_escapes
from math import fabs
import sys
from tkinter.tix import Tree
from matplotlib.pyplot import get

from tqdm import tqdm

sys.path.append("../..")

from typing import Dict, List, Union, Tuple
import torch
from models.token.data_classes import TokensBatch
from utils.training import configure_optimizers_alon
from utils.common import print_table
from utils.vocabulary import Vocabulary_token, SOS, PAD, UNK, EOS
from preprocess import get_data, symbolic, count_stmts_frequency, collect_vocab
from reference_model import BLSTM
from ICVH import ICVH
import numpy
from config import Config
from torch.utils.data import DataLoader, Dataset
from os.path import exists

from omegaconf import DictConfig
from utils.matrics import Statistic

from utils.converting import tokens_to_wrapped_numpy
from utils.vocabulary import Vocabulary_token
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

class TokenDataset(Dataset):
    _separator = " "
    def __init__(self, functions:List, config: Config,
                 vocabulary: Vocabulary_token):
        self.functions = functions
        self.config = config
        self._token_vocabulary = vocabulary.token_to_id

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, index) -> TokensSample:
        
        sample = self.functions[index]
        label = sample['target']
        tokens = sample['code_sym']
        # convert tokens to wrapped numpy array
        wrapped_tokens = tokens_to_wrapped_numpy(
            tokens,
            self._token_vocabulary,
            self.config.max_len,
            False,
        )
        n_tokens = min(self.config.max_len, len(tokens)) 

        return TokensSample(tokens=wrapped_tokens,
                            label=int(label),
                            n_tokens=n_tokens)

    def get_n_samples(self):
        return self._n_samples

def collate_wrapper(batch: List[TokensSample]) -> TokensBatch:
        return TokensBatch(batch)

def get_datalodar(dataset, batch_size, shuffle):
    datalodar = DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn = collate_wrapper, pin_memory=True)
    return datalodar

def train_reference_model(model, train_dataset, val_dataset, batch_size, epoch):
    train_dataloader = get_datalodar(train_dataset, batch_size, True)
    val_datalodaer = get_datalodar(val_dataset, batch_size, True)
    for index in tqdm(range(epoch)):
        train_outputs = []
        for batch_idx, batch in enumerate(train_dataloader):
            train_outputs.append(model.training_step(batch, batch_idx)) 
        val_outputs = []
        for batch_idx, batch in enumerate(val_datalodaer):
            val_outputs.append(model.validation_step(batch, batch_idx))   
        general_epoch_end(train_outputs, 'train')
        general_epoch_end(val_outputs, 'val') 
    torch.save(model, "data/reference.model")

def train_ICVH(ICVH, train_dataset, val_dataset, batch_size, epoch):
    train_dataloader = get_datalodar(train_dataset, batch_size, True)
    val_datalodaer = get_datalodar(val_dataset, batch_size, True)
    for index in tqdm(range(epoch)):
        train_outputs = []
        for batch_idx, batch in enumerate(train_dataloader):
            train_outputs.append(ICVH.training_step(batch, batch_idx)) 
        val_outputs = []
        for batch_idx, batch in enumerate(val_datalodaer):
            val_outputs.append(ICVH.validation_step(batch, batch_idx))   
        general_epoch_end(train_outputs, 'train')
        general_epoch_end(val_outputs, 'val') 
    torch.save(ICVH, "data/ICVH.model")


def general_epoch_end(outputs: List[Dict], group: str) -> Dict:
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"]
                                     for out in outputs]).mean().item()
            logs = {f"{group}/loss": mean_loss}
            logs.update(
                Statistic.union_statistics([
                    out["statistic"] for out in outputs
                ]).calculate_metrics(group)) 
            print(logs) 
if __name__=="__main__":
    with open('data/res.json', 'r') as f:
        res = json.load(f)
    freq_dict = count_stmts_frequency(res)
    collect_vocab(res)
    vocab = Vocabulary_token.load_vocabulary('data/vocab.pkl')
    config = Config()
    reference_model = BLSTM(config, vocab)
    size = len(res)
    val_dataset = TokenDataset(res[0:size//10], config, vocab)
    train_dataset = TokenDataset(res[size//10:], config, vocab) 
    # train_reference_model(reference_model, train_dataset, val_dataset, 16, 50)
    reference_model = torch.load('data/reference.model')
    
    ICVH = ICVH(config,  reference_model,vocab)

    train_ICVH(ICVH, train_dataset, val_dataset, 16, 50 )
      