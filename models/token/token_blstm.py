from typing import Dict, List, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam, Optimizer, SGD, Adamax, RMSprop

from models.token.data_classes import TokensBatch
from utils.training import configure_optimizers_alon
from utils.vocabulary import Vocabulary_token, SOS, PAD, UNK, EOS
from utils.matrics import Statistic
import numpy


class TOKEN_BLSTM(LightningModule):
    _negative_value = -numpy.inf
    _activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "lkrelu": nn.LeakyReLU(0.3)
    }
    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

    def __init__(
        self,
        config: DictConfig,
        vocabulary: Vocabulary_token,
    ):
        super().__init__()
        self._config = config
        self._vocabulary = vocabulary
        self.save_hyperparameters()

        self.init_layers()

    def _get_activation(self, activation_name: str) -> torch.nn.Module:
        if activation_name in self._activations:
            return self._activations[activation_name]
        raise KeyError(f"Activation {activation_name} is not supported")

    def _get_optimizer(self, optimizer_name: str) -> torch.nn.Module:
        if optimizer_name in self._optimizers:
            return self._optimizers[optimizer_name]
        raise KeyError(f"Optimizer {optimizer_name} is not supported")


    def init_layers(self):
        # embedding
        self.token_embedding = nn.Embedding(
            len(self._vocabulary.token_to_id),
            self._config.encoder.embedding_size,
            padding_idx=self._vocabulary.token_to_id[PAD])
        self.dropout_rnn = nn.Dropout(self._config.encoder.rnn_dropout)
        # BLSTM layer
        self.blstm_layer = nn.LSTM(
            input_size=self._config.encoder.embedding_size,
            hidden_size=self._config.encoder.rnn_size,
            num_layers=self._config.encoder.rnn_num_layers,
            bidirectional=self._config.encoder.use_bi_rnn,
            dropout=self._config.encoder.rnn_dropout
            if self._config.encoder.rnn_num_layers > 1 else 0,
        )
        # layer for attention
        # self.att_layer = LuongAttention(self.hidden_size)
        # self.att_layer = LocalAttention(self._config.classifier.hidden_size)

        # MLP
        layers = [
            nn.Linear(self._config.classifier.hidden_size,
                      self._config.classifier.hidden_size),
            self._get_activation(self._config.classifier.activation),
            nn.Dropout(0.5)
        ]
        if self._config.classifier.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({self._config.classifier.n_hidden_layers})"
            )
        for _ in range(self._config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(self._config.classifier.hidden_size,
                          self._config.classifier.hidden_size),
                self._get_activation(self._config.classifier.activation),
                nn.Dropout(0.5)
            ]
        self.hidden_layers = nn.Sequential(*layers)

        self.out_layer = nn.Linear(self._config.classifier.hidden_size, 2)

    # ===== OPTIMIZERS =====

    def configure_optimizers(
            self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return self._get_optimizer(self._config.hyper_parameters.optimizer)(
            self.parameters(), self._config.hyper_parameters.learning_rate)

    def forward(self, tokens: torch.Tensor,
                token_per_label: torch.Tensor) -> torch.Tensor:
        """
        :param tokens: (seq len; batch size)
        :param token_per_label: (batch size)
        :return: (batch size, output size)
        """

        batch_size = tokens.size(1)  # batch size: int
        # [batch size; seq len]
        # masks = tokens.new_zeros((batch_size, tokens.size(0)))
        # for i, n_tokens in enumerate(token_per_label):
        #     masks[i, n_tokens:] = self._negative_value
        # [seq len; batch size; embedding size]
        token_embeddings = self.token_embedding(tokens)

        # accelerating packing
        with torch.no_grad():
            is_contain_pad_id, first_pad_pos = torch.max(
                tokens == self._vocabulary.token_to_id[PAD], dim=0)
            first_pad_pos[~is_contain_pad_id] = tokens.shape[
                0]  # if no pad token use len+1 position
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos,
                                                           descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        token_embeddings = token_embeddings[:, sort_indices]

        packed_token_embeddings = nn.utils.rnn.pack_padded_sequence(
            token_embeddings, sorted_path_lengths)

        # packed_token_embeddings = nn.utils.rnn.pack_padded_sequence(
        #     token_embeddings, token_per_label, enforce_sorted=False)
        # (seq len; batch size; 2 * hidden size), h_n: (2, batch size, hidden size)
        _, (h_n, _) = self.blstm_layer(packed_token_embeddings)
        h_n = h_n.permute(1, 0, 2)  # (batch size; 2; hidden size)
        h_n = torch.sum(h_n, dim=1)  # (batch size; hidden size)
        h_n = self.dropout_rnn(h_n)[reverse_sort_indices]
        h_n = self.hidden_layers(h_n)  # (batch size; hidden size)
        out = self.out_layer(h_n)  # (batch size, output size)
        # out_prob = F.softmax(out.view(batch_size, -1)) # (batch size, output size)
        out_prob = torch.log_softmax(out.view(batch_size, -1),
                                     dim=1)  # (batch size, output size)

        return out_prob

    def training_step(self, batch: TokensBatch, batch_idx: int) -> Dict:
        # (batch size, output size)
        logits = self(batch.tokens, batch.tokens_per_label)
        loss = F.nll_loss(logits, batch.labels)
        log: Dict[str, Union[float, torch.Tensor]] = {"train/loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
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

    def validation_step(self, batch: TokensBatch, batch_idx: int) -> Dict:
        # (batch size, output size)
        logits = self(batch.tokens, batch.tokens_per_label)
        loss = F.nll_loss(logits, batch.labels)
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )

        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: TokensBatch, batch_idx: int) -> Dict:
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

    # ===== ON EPOCH END =====

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "test")
