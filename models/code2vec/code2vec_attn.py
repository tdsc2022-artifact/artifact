from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from models.code2seq.data_classes import PathContextBatch
from .path_encoder import PathEncoder
from .path_classifier import PathClassifier
from utils.training import configure_optimizers_alon
from utils.vocabulary import Vocabulary_c2s, PAD
from utils.matrics import Statistic


class Code2VecAttn(LightningModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary_c2s):
        super().__init__()
        self._config = config
        self.save_hyperparameters()
        self.encoder = PathEncoder(
            self._config.encoder,
            self._config.classifier.classifier_input_size,
            len(vocabulary.token_to_id),
            vocabulary.token_to_id[PAD],
            len(vocabulary.node_to_id),
            vocabulary.node_to_id[PAD],
        )
        self.num_classes = 2
        self.classifier = PathClassifier(self._config.classifier,
                                         self.num_classes)

    def configure_optimizers(
            self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return configure_optimizers_alon(self._config.hyper_parameters,
                                         self.parameters())

    def forward(self, samples: Dict[str, torch.Tensor],
                paths_for_label: List[int]) -> torch.Tensor:  # type: ignore
        return self.classifier(self.encoder(samples), paths_for_label)

    # ========== MODEL STEP ==========

    def training_step(self, batch: PathContextBatch,
                      batch_idx: int) -> Dict:  # type: ignore
        # [batch size; num_classes]
        logits = self(batch.contexts, batch.contexts_per_label)
        loss = F.cross_entropy(logits, batch.labels)
        log = {"train/loss": loss}
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

    def validation_step(self, batch: PathContextBatch,
                        batch_idx: int) -> Dict:  # type: ignore
        # [batch size; num_classes]
        logits = self(batch.contexts, batch.contexts_per_label)
        loss = F.cross_entropy(logits, batch.labels)
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )

        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: PathContextBatch,
                  batch_idx: int) -> Dict:  # type: ignore
        return self.validation_step(batch, batch_idx)

    # ========== ON EPOCH END ==========

    def _general_epoch_end(self, outputs: List[Dict], group: str):
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

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        return self._general_epoch_end(outputs, "test")
