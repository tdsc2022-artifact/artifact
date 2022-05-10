import os
import json
import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from torch_geometric.data import DataLoader
import torch
from torch_geometric.data import Data

#
# edge_index = torch.tensor([[0, 1],
#                            [2, 3],
#                            [4, 5],
#                            [6,7]
#                     ], dtype=torch.long)
# data = Data(edge_index=edge_index.t().contiguous())
from torch_geometric.data import DataLoader
# loader = DataLoader(data, batch_size=2, shuffle=True)
# for i in loader:
#     b = 1

import torch
from torch_geometric.data import InMemoryDataset
from gensim.test.utils import common_texts

class PGDataset(InMemoryDataset):
    def __init__(self, geo_dir,data_path,transform=None, pre_transform=None):
        
        self.data_path = data_path
        super(PGDataset, self).__init__(geo_dir, transform, pre_transform)
        
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

    def buildSensiXFGsDict(self):
        '''build Sensitive CFGs dict

        :param inputJsonDir:
        :return: {testcaseid:[{nodes:,nodes-vec:,edges:,target:},..]}
        '''
        SensiCFGsDict = dict()
        
        # print (fullpath)
        data_path = self.data_path
        curjson = []
        with open(data_path, 'r', encoding='utf8') as f:
            curjson = json.load(f)
            f.close()
        for cfgidx in range(len(curjson)):# {testcaseid:[{nodes-line-sym:,edges-No:,target:}]}
            cfg = curjson[cfgidx]
            if ("target" not in cfg.keys()):
                continue
            if 'testcaseID' in cfg.keys():
                tescase_id = cfg['testcaseID']
            else :
                tescase_id = str(cfgidx)
            sensiCFG = dict()

            sensiCFG["nodes"] = cfg["nodes-line-sym"]
            sensiCFG["edges"] = cfg["edges-No"]
            sensiCFG["target"] = int(cfg["target"])
            if tescase_id not in SensiCFGsDict.keys():
                SensiCFGsDict[tescase_id] = list()
                SensiCFGsDict[tescase_id].append(sensiCFG)
            else:
                SensiCFGsDict[tescase_id].append(sensiCFG)
                
                # sensiCFG["info"] = [int(file[:-5]),cfgidx]
                ## sensiCFG["lines"] = cfg["nodes-lineNo"]
        

        documents = list()
        print("START -- training doc2vec model......")

        for testcaseid in SensiCFGsDict:
            curSensiCFGs = SensiCFGsDict[testcaseid]
            for curSensiCFGsidx in range(len(curSensiCFGs)):
                curSensiCFG = curSensiCFGs[curSensiCFGsidx]
                curSensiCFGNodes = curSensiCFG["nodes"]
                for nodeidx in range(len(curSensiCFGNodes)):
                    node = curSensiCFGNodes[nodeidx]
                    documents.append(
                        TaggedDocument(node.split(), [testcaseid + "_" + str(curSensiCFGsidx) + "_" + str(nodeidx)]))

        model = Doc2Vec(documents, vector_size=64, min_count=5, workers=8, window=8, dm=0, alpha=0.025,
                        epochs=50)

        # for doc in labeled_corpus:
        #     words = filter(lambda x: x in model.vocab, doc.words)
        print("END -- training doc2vec model:{}!!!".format(model))

        for testcaseid in SensiCFGsDict:
            curSensiCFGs = SensiCFGsDict[testcaseid]
            for curSensiCFGsidx in range(len(curSensiCFGs)):
                curSensiCFG = curSensiCFGs[curSensiCFGsidx]
                curSensiCFGNodes = curSensiCFG["nodes"]
                nodeVecList = list()
                for nodeidx in range(len(curSensiCFGNodes)):
                    nodeVecList.append(model.docvecs[testcaseid + "_" + str(curSensiCFGsidx) + "_" + str(nodeidx)])
                curSensiCFG["nodes-vec"] = nodeVecList


        return SensiCFGsDict

    def process(self):
        # Read data into huge `Data` list.

        # data.x: Node feature matrix with shape [num_nodes, num_node_features]
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        # data.y: Target to train against (may have arbitrary shape)
        # data.pos: Node position matrix with shape [num_nodes, num_dimensions]

        print("START -- building sensitive CFGs Dict......")
        SensiCFGsDict = self.buildSensiXFGsDict()
        print("END -- building sensitive CFGs Dict!!!")


        # Read data into huge `Data` list.
        data_list = list()

        # TODO:解决样本不平衡的问题
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import RandomOverSampler
        from collections import Counter

        # smote_tomek = RandomOverSampler(random_state=0)
        X = list()
        Y = list()
        X_0 = list()
        X_1 = list()
        for testcaseid in SensiCFGsDict:
            curSensiCFGs = SensiCFGsDict[testcaseid]
            # X.append(curGraph)
            for curSensiCFG in curSensiCFGs:
                y = curSensiCFG["target"]
                if (y == 0):
                    X_0.append(curSensiCFG)
                else:
                    X_1.append(curSensiCFG)
        # num = abs(len(X_0) - len(X_1))
        # if (len(X_0) > len(X_1)):
        #     # 扩展x1
        #
        #     for i in range(num):
        #         X_1.append(random.choice(X_1))
        #
        # else:
        #     # 扩展x0
        #     for i in range(num):
        #         X_0.append(random.choice(X_0))

        X.extend(X_0)
        Y.extend(len(X_0) * [0])
        X.extend(X_1)
        Y.extend(len(X_1) * [1])

        # X_resampled, y_resampled = smote_tomek.fit_sample(X, Y)
        X_resampled, y_resampled = X,Y
        print("Samples distributions are follows:")
        print(sorted(Counter(y_resampled).items()))

        print("START -- building sensitive CFGs in pytorch_geometric DATA format......")
        # infoToData = dict()
        # for graphk in graphs:
        for curSensiCFG in X_resampled:
            # curGraph = graphs[graphk]
            # curGraph = curGraph[0]
            edge_index_v = curSensiCFG["edges"]

            x_v = curSensiCFG["nodes-vec"]
            y = torch.tensor([curSensiCFG["target"]], dtype=torch.long)

            x = torch.tensor(x_v, dtype=torch.float)
            if (len(edge_index_v) != 0):
                # consider each edge has the same attr
                # edge_attr = []
                # for i in range(len(edge_index_v)):
                #     edge_attr.append([1])
                # edge_attr = torch.tensor(edge_attr,dtype=torch.float)

                edge_index = torch.tensor(edge_index_v, dtype=torch.long)
                # data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y,edge_attr=edge_attr)
                data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
            else:
                # edge_attr = torch.tensor([], dtype=torch.float)
                edge_index = torch.tensor([], dtype=torch.long)
                # data = Data(edge_index=edge_index,x=x, y=y,edge_attr=edge_attr)
                data = Data(edge_index=edge_index,x=x, y=y)
            # info = torch.tensor(curSensiCFG["info"], dtype=torch.long)
            # data.info = info
            ## infoToData[curSensiCFG["info"]] = data
            ## print(edge_index.t().contiguous())
            ## print(data)
            data_list.append(data)
        random.shuffle(data_list)

        # edge_index = torch.tensor([[0, 1],
        #                            [1,2]
        #                            ], dtype=torch.long)
        # y1 = torch.tensor([0], dtype=torch.long)
        # y2 = torch.tensor([1], dtype=torch.long)
        #
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        # data1 = Data(x = x,edge_index=edge_index.t().contiguous(), y=y1)
        # data2 = Data(x=x,edge_index=edge_index.t().contiguous(), y=y2)
        #
        # data_list = [data1, data2, data1, data2]

        # edge_index = torch.tensor([[]
        #                            ], dtype=torch.long)
        # y1 = torch.tensor([0], dtype=torch.long)
        # y2 = torch.tensor([1], dtype=torch.long)
        #
        # x = torch.tensor([[-1]], dtype=torch.float)
        # data1 = Data(x=x, edge_index=edge_index.t().contiguous(), y=y1)
        # data2 = Data(x=x, edge_index=edge_index.t().contiguous(), y=y2)
        #
        # data_list = [data1, data2, data1, data2]

        # if self.pre_filter is not None:
        #     data_list [data for data in data_list if self.pre_filter(data)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("END -- building sensitive CFGs in pytorch_geometric DATA format!!!")
        print("START -- saving sensitive CFGs in pytorch_geometric DATA format......")

        torch.save((data, slices), self.processed_paths[0])
        print("END -- saving sensitive CFGs in pytorch_geometric DATA format!!!")



def main():
    cwe_id = '79'
    data_path = 'data/'
    geno_path = 'data/{}/'.format(cwe_id)
    dataset = PGDataset(data_path, geno_path, cwe_id)
    print(dataset)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # for i in loader:
    #     print(i)
    edge_index = torch.tensor([[0, 1],
                               ], dtype=torch.long)
    y = torch.tensor([0], dtype=torch.long)
    data1 = Data(edge_index=edge_index.t().contiguous(), y=y)
    
    print(data1)

if __name__ == '__main__':

    
    main()
