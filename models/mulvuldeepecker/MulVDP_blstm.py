from typing import Tuple, Dict, List, Union

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from omegaconf import DictConfig
from models.mulvuldeepecker.data_classes import MulVDPBatch
from utils.training import configure_optimizers_alon

import torch.nn as nn
import numpy
import torch.nn.functional as F
from utils.training import cut_sys_encoded_contexts
from utils.matrics import Statistic

from utils.vocabulary import Vocabulary_token
from torch.optim import RMSprop, Adam, Optimizer, SGD, Adamax


class MulVDP_BLSTM(LightningModule):
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
        self.g_blstm = nn.LSTM(
            input_size=self._config.global_encoder.embedding_size,
            hidden_size=self._config.global_encoder.rnn_size,
            num_layers=self._config.global_encoder.rnn_num_layers,
            bidirectional=self._config.global_encoder.use_bi_rnn,
            dropout=self._config.global_encoder.rnn_dropout
            if self._config.global_encoder.rnn_num_layers > 1 else 0,
            batch_first=True)
        self.l_blstm = nn.LSTM(
            input_size=self._config.local_encoder.embedding_size,
            hidden_size=self._config.local_encoder.rnn_size,
            num_layers=self._config.local_encoder.rnn_num_layers,
            bidirectional=self._config.local_encoder.use_bi_rnn,
            dropout=self._config.local_encoder.rnn_dropout
            if self._config.local_encoder.rnn_num_layers > 1 else 0,
            batch_first=True)

        self.g_dropout_rnn = nn.Dropout(
            self._config.global_encoder.rnn_dropout)
        self.l_dropout_rnn = nn.Dropout(
            self._config.global_encoder.rnn_dropout)

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

    def txt_rnn_embedding(self, txt: torch.Tensor, words_per_label: List[int],
                          op: str):
        '''
        encode txt using blstm

        :param txt: (total word length, input size)
        :param words_per_label: (batch size)
        :param op: "global" or "local"
        :return h_n: (batch size, hidden size)

        '''
        # x: (batch size, seq len, input size), masks: (batch size, seq len)
        x, masks = cut_sys_encoded_contexts(
            txt, words_per_label, self._config.hyper_parameters.seq_len,
            self._negative_value)
        lengths_per_label = [
            min(self._config.hyper_parameters.seq_len, word_per_label.item())
            for word_per_label in words_per_label
        ]
        # accelerating packing
        with torch.no_grad():
            first_pad_pos = torch.from_numpy(numpy.array(lengths_per_label))
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos,
                                                           descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        x = x[sort_indices]
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              sorted_path_lengths,
                                              batch_first=True)

        # (batch size, seq len, 2 * hidden size), h_n: (num layers * 2, batch size, hidden size)
        if op == "global":
            _, (h_n, _) = self.g_blstm(x)
        else:
            _, (h_n, _) = self.l_blstm(x)
        # (batch size, num layers * 2, hidden size)
        h_n = h_n.permute(1, 0, 2)
        h_n = torch.sum(h_n, dim=1)  # (batch size, hidden size)
        if op == "global":
            h_n = self.g_dropout_rnn(h_n)[reverse_sort_indices]
        else:
            h_n = self.l_dropout_rnn(h_n)[reverse_sort_indices]

        return h_n

    def forward(self, gadgets: torch.Tensor, global_words_per_label: List[int],
                attns: torch.Tensor,
                local_words_per_label: List[int]) -> torch.Tensor:
        """
        :param gadgets: (total word length, input size)
        :param words_per_label: word length for each label
        :return: (batch size, output size)
        """

        batch_size = len(global_words_per_label)
        g_x = self.txt_rnn_embedding(gadgets, global_words_per_label, "global")
        l_x = self.txt_rnn_embedding(attns, local_words_per_label, "local")
        # (batch size, hidden size)
        concat = torch.cat([g_x, l_x], dim=-1)

        out = self.out_layer(
            self.hidden_layers(concat))  # (batch size, output size)
        # out_prob = F.softmax(out.view(batch_size, -1)) # (batch size, output size)
        out_prob = torch.log_softmax(out.view(batch_size, -1),
                                     dim=1)  # (batch size, output size)

        return out_prob

    def training_step(self, batch: MulVDPBatch, batch_idx: int) -> Dict:
        # (batch size, output size)
        logits = self(batch.gadgets, batch.global_tokens_per_label,
                      batch.attns, batch.local_tokens_per_label)
        # loss = F.cross_entropy(logits, batch.labels)
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

    def validation_step(self, batch: MulVDPBatch, batch_idx: int) -> Dict:
        # (batch size, output size)
        logits = self(batch.gadgets, batch.global_tokens_per_label,
                      batch.attns, batch.local_tokens_per_label)

        loss = F.nll_loss(logits, batch.labels)
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )

        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: MulVDPBatch, batch_idx: int) -> Dict:
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
        return self._get_optimizer(self._config.hyper_parameters.optimizer)(
            self.parameters(), self._config.hyper_parameters.learning_rate)

    # ===== ON EPOCH END =====

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "test")
