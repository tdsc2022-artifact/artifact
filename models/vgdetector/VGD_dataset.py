'''build pytorch_geometric dataset

@author : jumormt
@version : 1.0
'''
import json
import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class VGDDataset(InMemoryDataset):
    def __init__(self,
                 geo_path,
                 data_path,
                 vector_length,
                 transform=None,
                 pre_transform=None):
        self._data_path = data_path
        self._geo_path = geo_path
        self._vector_length = vector_length
        super(VGDDataset, self).__init__(geo_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']
        # pass

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def vectorize_cfg_nodes(self, cfg_json):
        '''vectorize cfg nodes

        :param cfg_json:
        :return: [cfgs]
        '''
        print("START -- process {}......".format(cfg_json))

        with open(cfg_json, 'r', encoding="utf-8") as f:
            cfgs = json.load(f)
            documents = list()
            print("START -- training doc2vec model......")

            for cfg_idx in range(len(cfgs)):
                cfg = cfgs[cfg_idx]
                cfg_nodes = cfg["nodes"]
                for node_idx in range(len(cfg_nodes)):
                    node = cfg_nodes[node_idx]
                    documents.append(
                        TaggedDocument(node,
                                       [str(cfg_idx) + "_" + str(node_idx)]))

            model = Doc2Vec(documents,
                            vector_size=self._vector_length,
                            min_count=5,
                            workers=8,
                            window=8,
                            dm=0,
                            alpha=0.025,
                            epochs=50)

            # for doc in labeled_corpus:
            #     words = filter(lambda x: x in model.vocab, doc.words)
            print("END -- training doc2vec model:{}!!!".format(model))

            for cfg_idx in range(len(cfgs)):
                cfg = cfgs[cfg_idx]
                nodes = cfg["nodes"]
                nodeVecList = list()
                for node_idx in range(len(nodes)):
                    nodeVecList.append(model.docvecs[str(cfg_idx) + "_" +
                                                     str(node_idx)])
                cfg["nodes-vec"] = nodeVecList

        return cfgs

    def process(self):
        print("START -- building sensitive CFGs Dict......")
        cfgs = self.vectorize_cfg_nodes(self._data_path)
        print("END -- building sensitive CFGs Dict!!!")

        # Read data into huge `Data` list.
        data_list = list()
        X = list()
        Y = list()
        X_0 = list()
        X_1 = list()

        # X.append(curGraph)
        for cfg in cfgs:
            y = cfg["target"]
            if (y == 0):
                X_0.append(cfg)
            else:
                X_1.append(cfg)

        X.extend(X_0)
        Y.extend(len(X_0) * [0])
        X.extend(X_1)
        Y.extend(len(X_1) * [1])

        print(
            "START -- building sensitive CFGs in pytorch_geometric DATA format......"
        )
        for cfg in X:
            edge_index_v = cfg["edges"]

            x_v = cfg["nodes-vec"]
            y = torch.tensor([cfg["target"]], dtype=torch.long)

            x = torch.tensor(x_v, dtype=torch.float)
            if (len(edge_index_v) != 0):

                edge_index = torch.tensor(edge_index_v, dtype=torch.long)
                data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, metric_info=[cfg['sp'], cfg['sd'], cfg['sp&sd'], cfg['sp|sd']])
            else:
                edge_index = torch.tensor([], dtype=torch.long)
                data = Data(edge_index=edge_index, x=x, y=y, metric_info=[cfg['sp'], cfg['sd'], cfg['sp&sd'], cfg['sp|sd']])

            data_list.append(data)
        random.shuffle(data_list)

        data, slices = self.collate(data_list)
        print(
            "END -- building sensitive CFGs in pytorch_geometric DATA format!!!"
        )
        print(
            "START -- saving sensitive CFGs in pytorch_geometric DATA format......"
        )

        torch.save((data, slices), self.processed_paths[0])
        print(
            "END -- saving sensitive CFGs in pytorch_geometric DATA format!!!")
