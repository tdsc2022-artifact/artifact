from typing import Dict, List, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam, Optimizer, SGD, Adamax, RMSprop
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from models.token.data_classes import TokensBatch
from utils.training import configure_optimizers_alon
from utils.vocabulary import Vocabulary_token, SOS, PAD, UNK, EOS
import numpy
from utils.matrics import Statistic


class BLSTM(nn.Module):

    def __init__(
        self,
        config,
        vocabulary: Vocabulary_token,
    ):
        super().__init__()
        self._config = config
        self._vocabulary = vocabulary
        self.init_layers()
        
    def init_layers(self):
        # embedding
        self.token_embedding = nn.Embedding(
            len(self._vocabulary.token_to_id),
            self.embedding_size,
            padding_idx = self._vocabulary.token_to_id[PAD])
        self.relu = nn.ReLU()
        self.optimizer = Adam(self.parameters(),
                         self._config.learning_rate,
                         self._config.weight_decay)
       
        # BLSTM layer
        self.blstm_layer = nn.LSTM(
            input_size = self._config.blstm_input_size,
            hidden_size = self._config.blstm_hidden_size,
            num_layers = self._config.blstm_num_layers,
            bidirectional = True,
        )
        # layer for attention
        # self.att_layer = LuongAttention(self.hidden_size)
        # self.att_layer = LocalAttention(self._config.classifier.hidden_size)
        self.dropout_rnn = nn.Dropout(0.5)
        # MLP
        layers = [
            nn.Linear(self._config.mlp_hidden_size,
                      self._config.mlp_hidden_size),
            self.relu,
            nn.Dropout(0.5)
        ]
        for _ in range(2):
            layers += [
                nn.Linear(self._config.mlp_hidden_size,
                          self._config.mlp_hidden_size),
                self.relu,
                nn.Dropout(0.5)
            ]
        self.hidden_layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(self._config.mlp_hidden_size, 2)
        self.device =  torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.cuda(0)
        self.to(self.device)
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
        if torch.cuda.is_available():
            batch.move_to_device(self.device)

        self.optimizer.zero_grad()
        logits = self(batch.tokens, batch.tokens_per_label)
        loss = F.nll_loss(logits, batch.labels)
        loss.backward()
        self.optimizer.step()

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
        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: TokensBatch, batch_idx: int) -> Dict:
        # (batch size, output size)
        if torch.cuda.is_available():
            batch.move_to_device(self.device)

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

    def predict(self, batch: TokensBatch):
        logits = self(batch.tokens, batch.tokens_per_label)
        _, preds = logits.max(dim=1)

        return preds

    
            
    