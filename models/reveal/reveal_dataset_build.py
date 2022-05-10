import json
import shutil
from gensim.models.word2vec import Word2Vec

import numpy as np
from omegaconf.dictconfig import DictConfig
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset, DataLoader, Batch

import random
import os
from models.reveal.REVEAL_ggnn import type_map
import nltk

random.seed(7)




def vectorize_single_statement(config, node_content: str, pretrain_word2vec_model, node_type: str):
    
    '''
    向量化1个statement
    :param node_content: 一个statement的内容，比如'int a = 10; '
    :param node_type: statement类型，比如ExpressionStatement
    :return: 该statement向量
    '''
    
    node_split = nltk.word_tokenize(node_content)
    nrp = np.zeros(config.hyper_parameters.vector_length)  # 100
    
    type_one_hot = np.eye(len(type_map))
    for token in node_split:
        try:
            embedding = pretrain_word2vec_model.wv[token]
        except:
            embedding = np.zeros(config.hyper_parameters.vector_length)
        nrp = np.add(nrp, embedding)
    if len(node_split) > 0:
        fNrp = np.divide(nrp, len(node_split))
    else:
        fNrp = nrp
    node_feature = type_one_hot[type_map[node_type] - 1].tolist()
    node_feature.extend(fNrp.tolist())
    
    return node_feature  # dim = 169

class RevealDataset(InMemoryDataset):
    '''
    json结构:
    - node-line-content
    - statement-type
    - control_flow_edge
    - control_dependency_edge
    - data_dependency_edge
    '''
    def __init__(self, config:DictConfig, geo_path, data_json, transform=None, pre_transform=None):
        self._geo_path = geo_path
        self._config = config
        self.data_json = data_json
        self._dataset_dir = os.path.join(self._config.data_folder, self._config.name, self._config.dataset.name)  # 数据集文件路径，可以自定义
        if(os.path.exists(self._geo_path)):
            shutil.rmtree(self._geo_path)
        super(RevealDataset, self).__init__(self._geo_path, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']




    def vectorize_all_cpg(self, config:DictConfig, data_json:list):
        '''
        向量化数据集中所有的cpg
        :param positive_cpgs: 带漏洞的cpg
        :param negative_cpgs: benign cpg
        :return: 所有向量化后的cpg
        '''
        w2v_path = os.path.join(config.data_folder, config.name, config.dataset.name, f'w2v_model/{config.dataset.name}.w2v')
        pretrain_word2vec_model = Word2Vec.load(w2v_path)
        count = 0
        for cpg_idx in range(len(data_json)):
            cpg = data_json[cpg_idx]
            cpg_node_contents = cpg["node-line-content"]  # 获取CPG结点的内容
            cpg_node_types = cpg["statement-type"]
            nodeVecList = [vectorize_single_statement(config, content, pretrain_word2vec_model, type) for content, type in zip(cpg_node_contents, cpg_node_types)]
            cpg["nodes-vec"] = nodeVecList
            count += 1
            print(f'vectoring {count} cpgs', end='\r')

        return data_json

    def process(self):
        print("START -- building sensitive CPGs Dict......")

        cpgs = self.vectorize_all_cpg(self._config, self.data_json)
        print("END -- building sensitive CPGs Dict!!!")

        # Read data into huge `Data` list.
        data_list = list()

        # X.append(curGraph)
        print("START -- building sensitive CPGs in pytorch_geometric DATA format......")

        for cpg in cpgs:
            edge_index_v = []
            for edge_type in ["control_flow_edge", "control_dependency_edge", "data_dependency_edge"]:
                edge_index_v.extend(cpg[edge_type])

            x_v = cpg["nodes-vec"]
            y = torch.tensor([cpg["target"]], dtype=torch.long)

            x = torch.tensor(x_v, dtype=torch.float)
            if (len(edge_index_v) != 0):
                edge_index = torch.tensor(edge_index_v, dtype=torch.long)
                data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, flip=cpg['flip'], xfg_id=cpg['xfg_id'])
            else:
                edge_index = torch.tensor([], dtype=torch.long)
                data = Data(edge_index=edge_index, x=x, y=y, flip=cpg['flip'], xfg_id=cpg['xfg_id'])

            data_list.append(data)

        data, slices = self.collate(data_list)
        print("END -- building sensitive CPGs in pytorch_geometric DATA format!!!")
        print("START -- saving sensitive CPGs in pytorch_geometric DATA format......")

        torch.save((data, slices), self.processed_paths[0])
        print("END -- saving sensitive CPGs in pytorch_geometric DATA format!!!")

