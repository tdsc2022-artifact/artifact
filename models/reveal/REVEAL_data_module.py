from builtins import slice
from os.path import exists, join
import shutil
from typing import List, Optional, Tuple
import random
import numpy as np
from numpy.lib.function_base import flip
from sklearn import preprocessing
from utils.json_ops import read_json
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
import sys
sys.path.append('..')
from torch_geometric.data import DataLoader
from utils.json_ops import write_json, read_json
from torch.utils.data import Subset
from torch_geometric.data import  DataLoader, Batch
import os
from utils.vocabulary import Vocabulary_token
from models.reveal.reveal_dataset_build import RevealDataset
import json
class RevealDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary_token):
        super().__init__()
        self._config = config
        self._dataset_dir = os.path.join(self._config.data_folder, 
                                         self._config.name,
                                         self._config.dataset.name
                                         )
        self.data_path = os.path.join(self._dataset_dir, f'{self._config.dataset.name}.json')  
        self.data_json = read_json(self.data_path) 
        self._geo_dir = join(self._config.geo_folder, self._config.name, self._config.dataset.name, 'geo')
        #处理数据
        
        
    def get_data_class_rate(self):
        
        negative_cpgs = 0
        positive_cpgs = 0
        for cpg in self.train_xfgs:
            if(cpg['target'] == 1):
                positive_cpgs += 1
            else:
                negative_cpgs += 1
        
        return negative_cpgs / positive_cpgs, len(self.test_xfgs)
    
     
        
        
    def prepare_data(self):
        
        print('prepare_data')
        
        size = len(self.data_json) 
        train_slice = slice(size // 5, len(self.data_json))
        val_slice = slice(0, size // 5)
        test_slice = slice(size // 10, size // 5)
        self.train_xfgs = self.data_json[train_slice]
        self.val_xfgs = self.data_json[val_slice]
        self.test_xfgs = self.data_json[test_slice]
        self.data_class_rate, self.test_n_samples = self.get_data_class_rate()  
        self.train_dataset = RevealDataset(self._config, self._geo_dir, self.train_xfgs)
        self.val_dataset = RevealDataset(self._config, self._geo_dir, self.val_xfgs)
        self.test_dataset = RevealDataset(self._config, self._geo_dir, self.test_xfgs)
        
        
        
        # TODO: download data from s3 if not exists

    def create_dataloader(
        self,
        dataset: RevealDataset,
        shuffle: bool,
        batch_size: int,
        n_workers: int,
    ) -> Tuple[DataLoader, int]:
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=n_workers,
            pin_memory=True,
        )
        return dataloader, len(dataset)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        train_dataset = self.train_dataset
        train_dataloader, _ = self.create_dataloader(train_dataset,
                                      self._config.hyper_parameters.shuffle_data, 
                                      self._config.hyper_parameters.batch_size,
                                      self._config.num_workers)
        return train_dataloader

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        # val_dataset = Subset(self.dataset, self.val_slice)
        val_dataset = self.val_dataset
        val_dataloader, _ = self.create_dataloader(val_dataset,
                                      self._config.hyper_parameters.shuffle_data, 
                                      self._config.hyper_parameters.test_batch_size,
                                      self._config.num_workers)
        return val_dataloader

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        # test_dataset = Subset(self.dataset, self.test_slice)
        test_dataset = self.test_dataset
        test_dataloader, _ = self.create_dataloader(test_dataset,
                                      self._config.hyper_parameters.shuffle_data, 
                                      self._config.hyper_parameters.test_batch_size,
                                      self._config.num_workers)
        return test_dataloader

    def transfer_batch_to_device(self, batch, device: torch.device):
        batch.to(device)
        return batch


# 尽量保证train, test, val中每部分positive和negative sample比例一样
def shuffle_datas_indices(positive_cpgs: list, negative_cpgs: list):
    positive_cpg_indices = list(range(len(positive_cpgs)))
    negative_cpg_indices = list(range(len(negative_cpgs)))
    random.shuffle(positive_cpg_indices)
    random.shuffle(negative_cpg_indices)

    sz = len(positive_cpg_indices)
    positive_train_dataset_slice = slice(sz // 5, sz)  # 20% - 100%作为训练集
    positive_val_dataset_slice = slice(0, sz // 10)  # 0 - 10%作为验证集
    positive_test_dataset_slice = slice(sz // 10, sz // 5)  # 10% - 20%作为验证集

    train_positive_cpgs_indices = positive_cpg_indices[positive_train_dataset_slice]
    val_positive_cpgs_indices = positive_cpg_indices[positive_val_dataset_slice]
    test_positive_cpgs_indices = positive_cpg_indices[positive_test_dataset_slice]

    sz = len(negative_cpg_indices)
    negative_train_dataset_slice = slice(sz // 5, sz)  # 20% - 100%作为训练集
    negative_val_dataset_slice = slice(0, sz // 10)  # 0 - 10%作为验证集
    negative_test_dataset_slice = slice(sz // 10, sz // 5)  # 10% - 20%作为验证集

    train_negative_cpgs_indices = negative_cpg_indices[negative_train_dataset_slice]
    val_negative_cpgs_indices = negative_cpg_indices[negative_val_dataset_slice]
    test_negative_cpgs_indices = negative_cpg_indices[negative_test_dataset_slice]

    return len(positive_cpgs), len(negative_cpgs),train_positive_cpgs_indices, val_positive_cpgs_indices, test_positive_cpgs_indices, \
           train_negative_cpgs_indices, val_negative_cpgs_indices, test_negative_cpgs_indices


# Triple Loss辅助函数
def generate_idx(idx = -1, length = 0):
    num = random.randint(0, length - 1)
    if idx != -1:
        while num == idx:
            num = random.randint(0, length - 1)
    return num


def generate_batch(datasets: list, batch_size: int):
    num_batch = len(datasets) // batch_size
    if len(datasets) % batch_size != 0:
        num_batch += 1

    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(datasets))

        a_data = [data[0] for data in datasets[start_idx: end_idx]]
        p_data = [data[1] for data in datasets[start_idx: end_idx]]
        n_data = [data[2] for data in datasets[start_idx: end_idx]]

        yield Batch.from_data_list(a_data), Batch.from_data_list(p_data), Batch.from_data_list(n_data)

# Triple Loss
def get_dataloader_triple_loss(config, dataset, dataset_dir):
    
    positive_cpgs = [cpg for cpg in json.load(open(f'{dataset_dir}/positive.json', 'r', encoding='utf-8'))
                         if len(cpg['node-line-content']) > 3]
    negative_cpgs = [cpg for cpg in json.load(open(f'{dataset_dir}/negative.json', 'r', encoding='utf-8'))
                         if len(cpg['node-line-content']) > 3 and 'bad' not in cpg['functionName']]
    
    len_positive, len_negative, train_positive_cpgs_indices, val_positive_cpgs_indices, test_positive_cpgs_indices, \
    train_negative_cpgs_indices, val_negative_cpgs_indices, test_negative_cpgs_indices = shuffle_datas_indices(
        positive_cpgs, negative_cpgs)

    val_slice = val_positive_cpgs_indices + [indice + len_positive for indice in val_negative_cpgs_indices]
    test_slice = test_positive_cpgs_indices + [indice + len_positive for indice in test_negative_cpgs_indices]
    valset = Subset(dataset, val_slice)
    testset = Subset(dataset, test_slice)

    dataloader = dict()
    dataloader['eval'] = DataLoader(valset, batch_size=config.hyper_parameters.test_batch_size, shuffle=config.hyper_parameters.shuffle_data)
    dataloader['test'] = DataLoader(testset, batch_size=config.hyper_parameters.test_batch_size, shuffle=False)

    # 构造训练三元组
    train_datas = []
    for i, cpg_idx in enumerate(train_positive_cpgs_indices):
        anchor_data = dataset[cpg_idx]

        p_idx = generate_idx(i, len(train_positive_cpgs_indices))
        n_idx = generate_idx(-1, len(train_negative_cpgs_indices))
        p_data = dataset[train_positive_cpgs_indices[p_idx]]
        n_data = dataset[len_positive + train_negative_cpgs_indices[n_idx]]

        train_datas.append((anchor_data, p_data, n_data))

    for i, cpg_idx in enumerate(train_negative_cpgs_indices):
        anchor_data = dataset[cpg_idx]

        p_idx = generate_idx(i, len(train_negative_cpgs_indices))
        n_idx = generate_idx(-1, len(train_positive_cpgs_indices))
        p_data = dataset[len_positive + train_negative_cpgs_indices[p_idx]]
        n_data = dataset[train_positive_cpgs_indices[n_idx]]

        train_datas.append((anchor_data, p_data, n_data))

    train_loader = generate_batch(train_datas, batch_size=config.hyper_parameters.batch_size)

    dataloader['train'] = train_loader
 
    
    return dataloader





# 交叉熵损失
def get_dataloader(config, dataset, dataset_dir):
    
    positive_cpgs = [cpg for cpg in json.load(open(f'{dataset_dir}/positive.json', 'r', encoding='utf-8'))
                         if len(cpg['node-line-content']) > 3]
    negative_cpgs = [cpg for cpg in json.load(open(f'{dataset_dir}/negative.json', 'r', encoding='utf-8'))
                         if len(cpg['node-line-content']) > 3 and 'bad' not in cpg['functionName']]
    
    len_positive, len_negative, train_positive_cpgs_indices, val_positive_cpgs_indices, test_positive_cpgs_indices, \
    train_negative_cpgs_indices, val_negative_cpgs_indices, test_negative_cpgs_indices = shuffle_datas_indices(positive_cpgs, negative_cpgs)

    train_slice = train_positive_cpgs_indices + [indice + len_positive for indice in train_negative_cpgs_indices]
    val_slice = val_positive_cpgs_indices + [indice + len_positive for indice in val_negative_cpgs_indices]
    test_slice = test_positive_cpgs_indices + [indice + len_positive for indice in test_negative_cpgs_indices]

    trainset = Subset(dataset, train_slice)
    valset = Subset(dataset, val_slice)
    testset = Subset(dataset, test_slice)


    dataloader = dict()
    dataloader['train'] = DataLoader(trainset, batch_size=config.hyper_parameters.batch_size, shuffle=config.hyper_parameters.shuffle_data)
    dataloader['eval'] = DataLoader(valset, batch_size=config.hyper_parameters.test_batch_size, shuffle=config.hyper_parameters.shuffle_data)
    dataloader['test'] = DataLoader(testset, batch_size=config.hyper_parameters.test_batch_size, shuffle=False)

    return dataloader

if __name__ == '__main__':
    dataset = RevealDataset()
    positive_cpgs = [cpg for cpg in json.load(open(f'{data_args.dataset_dir}/positive.json', 'r', encoding='utf-8')) if len(cpg['node-line-content']) > 3]
    negative_cpgs = [cpg for cpg in json.load(open(f'{data_args.dataset_dir}/negative.json', 'r', encoding='utf-8')) if len(cpg['node-line-content']) > 3
                     and 'bad' not in cpg['functionName']]
    dataloader = get_dataloader_triple_loss(dataset, positive_cpgs, negative_cpgs)

    train_loader = dataloader['train']

    for datas in train_loader:
        print('===================')
        print(datas[0])
        print(datas[1])
        print(datas[2])