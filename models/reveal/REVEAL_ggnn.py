import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.data import Batch, Data
from omegaconf import DictConfig
from typing import Tuple, Dict, List, Union
import torch.nn.functional as F
from utils.matrics import Statistic
from utils.training import configure_optimizers_alon
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim import Adam, Optimizer, SGD, Adamax
from utils.vocabulary import Vocabulary_token
type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostIncDecOperationExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69
}
class ClassifyModel(LightningModule):
    def __init__(self, config, vocabulary:Vocabulary_token):
        super().__init__()
        self._config = config
        self._MLP_internal_dim = int(self._config.classifier.hidden_size / 2)
        self._input_dim = len(type_map) + self._config.hyper_parameters.vector_length
        # GGNN层
        self.GGNN = GatedGraphConv(out_channels=self._input_dim, num_layers=self._config.ggnn.layer_num)
        self.dropout_p = self._config.classifier.drop_out

        # MLP层
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self._input_dim, out_features=self._config.classifier.hidden_size, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=self._config.classifier.hidden_size, out_features=self._MLP_internal_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features=self._MLP_internal_dim, out_features=self._config.classifier.hidden_size, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        ) for _ in range(self._config.classifier.n_hidden_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self._config.classifier.hidden_size, out_features=2),
        )
        rate = 1
        weight_ce = torch.FloatTensor([1, rate]).to("cuda:{}".format(config.gpu) if torch.cuda.is_available() else "cpu")
        self.loss_function = nn.CrossEntropyLoss(weight=weight_ce)


    def extract_feature(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_emb = self.GGNN(x, edge_index)
        graph_emb = self.GGNN.aggregate(node_emb, index=batch) # [batch_size, embedding_dim]

        feature_emb = self.extract_feature(graph_emb)
        probs = self.classifier(feature_emb) # [batch_size, 2]
        return probs

    def training_step(self, batch, batch_idx: int) -> Dict:
        # (batch size, output size)
        # logits = self(batch)
        logits = self(batch)
        # logits = F.log_softmax(x, dim=-1)
        loss = self.loss_function(logits, batch.y)
        # loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=2)
        # loss = loss_fn(x, batch.y)
        log: Dict[str, Union[float, torch.Tensor]] = {"train/loss": loss}


        with torch.no_grad():
            _, preds = logits.max(dim=1)
            # with open('srad_simple_analysis/train_false_xfg_ids.txt', 'a+') as f:
            #     for xfg_id in batch.xfg_id[batch.y != preds]:
            #         f.write(str(xfg_id.tolist()) + ',')
            #     f.close()
            statistic = Statistic().calculate_statistic(
                batch.y,
                preds,
                2,
            )
            batch_matric = statistic.calculate_metrics(group="train")
            log.update(batch_matric)
            self.log_dict(log)
            self.log("f1",
                     batch_matric["train/f1"],
                     prog_bar=True,
                     logger=False)

        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch, batch_idx: int) -> Dict:
        # (batch size, output size)
        # logits = self(batch)
        logits = self(batch)
        # logits = F.log_softmax(x, dim=-1)
        # loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=2)
        loss = self.loss_function(logits, batch.y)
        # loss = loss_fn(x, batch.y)
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            # with open('srad_simple_analysis/val_false_xfg_ids.txt', 'a+') as f:
            #     for xfg_id in batch.xfg_id[batch.y != preds]:
            #         f.write(str(xfg_id.tolist()) + ',')
            #     f.close()
            statistic = Statistic().calculate_statistic(
                batch.y,
                preds,
                2,
            )

        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch, batch_idx: int) -> Dict:
                # (batch size, output size)
        # logits = self(batch)
        logits = self(batch)
        # logits = F.log_softmax(x, dim=-1)
        # loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=2)
        loss = self.loss_function(logits, batch.y)
        # loss = loss_fn(x, batch.y)
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            # with open('srad_simple_analysis/test_false_xfg_ids.txt', 'a+') as f:
            #     for xfg_id in batch.xfg_id[batch.y != preds]:
            #         f.write(str(xfg_id.tolist()) + ',')
            #     f.close()
            statistic = Statistic().calculate_statistic(
                batch.y,
                preds,
                2,
            )

        return {"loss": loss, "statistic": statistic}

    def _general_epoch_end(self, outputs: List[Dict], group: str) -> Dict:
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"]
                                     for out in outputs]).mean().item()
            logs = {f"{group}/loss": mean_loss}
            logs.update(
                Statistic.union_statistics([
                    out["statistic"] for out in outputs
                ]).calculate_metrics(group))
            self.log_dict(logs)
            self.log(f"{group}_loss", mean_loss)

    # ===== OPTIMIZERS =====

    def configure_optimizers(
            self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = Adam(self.parameters(), lr=self._config.hyper_parameters.learning_rate,
                         weight_decay=self._config.hyper_parameters.weight_decay,
                         betas=(self._config.hyper_parameters.beta1, self._config.hyper_parameters.beta2))
        scheduler = LambdaLR(
        optimizer, lr_lambda=lambda epoch: self._config.hyper_parameters.decay_gamma**epoch)
        return [optimizer], [scheduler]

    # ===== ON EPOCH END =====

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "test")
if __name__ == '__main__':
    input_dim = 169
    num_node = 5
    # model = GatedGraphConv(out_channels=input_dim, num_layers=5)
    from argparse import ArgumentParser
    import sys
    sys.path.append('/home/niexu/project/python/noise_reduce/')
    from utils.common import get_config
    arg_parser = ArgumentParser()
    # arg_parser.add_argument("model", type=str)
    # arg_parser.add_argument("--dataset", type=str, default=None)
    arg_parser.add_argument("--offline", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    args = arg_parser.parse_args()
    config = get_config('CWE119', 'reveal' ,log_offline=args.offline)
    model = ClassifyModel(config=config)

    x = torch.FloatTensor(num_node, input_dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 0], [1, 2, 3, 4, 4, 2]], dtype=torch.long)
    data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
    probs = model(data)

    print(probs.size())




