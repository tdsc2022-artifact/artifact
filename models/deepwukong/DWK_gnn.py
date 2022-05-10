from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from typing import Tuple, Dict, List, Union

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from omegaconf import DictConfig
from utils.training import configure_optimizers_alon
from utils.vocabulary import Vocabulary_token
import torch.nn as nn
import torch.nn.functional as F
from utils.matrics import Statistic



class GCNPoolBlockLayer(torch.nn.Module):
    """graph conv-pool block

    graph convolutional + graph pooling + graph readout

    :attr GCL: graph conv layer
    :attr GPL: graph pooling layer
    """
    def __init__(self, config: DictConfig):
        super(GCNPoolBlockLayer, self).__init__()
        self._config = config
        input_size = self._config.hyper_parameters.vector_length
        self.layer_num = self._config.gnn.layer_num
        self.input_GCL = GCNConv(input_size, config.gnn.hidden_size)

        self.input_GPL = TopKPooling(config.gnn.hidden_size,
                                     ratio=config.gnn.pooling_ratio)

        for i in range(self.layer_num - 1):
            setattr(self, f"hidden_GCL{i}",
                    GCNConv(config.gnn.hidden_size, config.gnn.hidden_size))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(config.gnn.hidden_size,
                            ratio=config.gnn.pooling_ratio))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.input_GCL(x, edge_index))
        x, edge_index, _, batch, _, _ = self.input_GPL(x, edge_index, None,
                                                       batch)
        # (batch size, hidden)
        out = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        for i in range(self.layer_num - 1):
            x = F.relu(getattr(self, f"hidden_GCL{i}")(x, edge_index))
            x, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                x, edge_index, None, batch)
            out += torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        return out


class DWK_GNN(LightningModule):
    _activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "lkrelu": nn.LeakyReLU(0.3)
    }

    def __init__(
        self,
        config: DictConfig,
        vocabulary: Vocabulary_token
    ):
        super().__init__()
        self._config = config
        self.save_hyperparameters()
        self.vocabulary = Vocabulary_token
        self.init_layers()

    def init_layers(self):
        self.gnn_layer = GCNPoolBlockLayer(self._config)
        self.lin1 = nn.Linear(self._config.gnn.hidden_size * 2,
                              self._config.gnn.hidden_size)
        self.dropout1 = nn.Dropout(self._config.classifier.drop_out)
        self.lin2 = nn.Linear(self._config.gnn.hidden_size,
                              self._config.gnn.hidden_size // 2)
        self.dropout2 = nn.Dropout(self._config.classifier.drop_out)
        self.lin3 = nn.Linear(self._config.gnn.hidden_size // 2, 2)

    def _get_activation(self, activation_name: str) -> torch.nn.Module:
        if activation_name in self._activations:
            return self._activations[activation_name]
        raise KeyError(f"Activation {activation_name} is not supported")

    def forward(self, batch):
        # (batch size, hidden)
        x = self.gnn_layer(batch)
        act = self._get_activation(self._config.classifier.activation)
        x = self.dropout1(act(self.lin1(x)))
        x = self.dropout2(act(self.lin2(x)))
        # (batch size, output size)
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x

    def training_step(self, batch, batch_idx: int) -> Dict:
        # (batch size, output size)
        logits = self(batch)

        loss = F.nll_loss(logits, batch.y)
        log: Dict[str, Union[float, torch.Tensor]] = {"train/loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
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
        logits = self(batch)

        loss = F.nll_loss(logits, batch.y)
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.y,
                preds,
                2,
            )

        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch, batch_idx: int) -> Dict:
        return self.validation_step(batch, batch_idx)

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
        return configure_optimizers_alon(self._config.hyper_parameters,
                                         self.parameters())

    # ===== ON EPOCH END =====

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "test")
