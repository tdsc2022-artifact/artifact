from os.path import join, split, exists
import os
from typing import List, Dict
import time
from pytorch_lightning import Callback, LightningModule, Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger

from utils.common import print_table
from omegaconf import DictConfig
import json


class UploadCheckpointCallback(Callback):
    def __init__(self, checkpoint_dir: str):
        super().__init__()
        self._checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            experiment = logger.experiment
            root_dir, _ = split(self._checkpoint_dir)
            experiment.save(join(self._checkpoint_dir, "*.ckpt"),
                            base_path=root_dir)


class PrintEpochResultCallback(Callback):
    def __init__(self, *groups: str):
        self._groups = groups

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        metrics_to_print: Dict[str, List[str]] = {
            group: []
            for group in self._groups
        }
        for key, value in trainer.callback_metrics.items():
            if "/" not in key:
                continue
            group, metric = key.split("/")
            if group in metrics_to_print:
                metrics_to_print[group].append(f"{metric}={round(value, 2)}")
        print_table(metrics_to_print)


class CollectResCallback(Callback):
    def __init__(self, config: DictConfig, data_module: LightningDataModule):
        self._config = config
        self.data_module = data_module

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        self.train_start_time = time.time()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        self.train_end_time = time.time()

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule):
        self.start_test_time = time.time()

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        self.end_test_time = time.time()
        json_out = join("res", f"{self._config.name}.json")
        if (not exists(json_out)):
            os.system(f"touch {json_out} && echo '{{}}' > {json_out}")
        res = {}
        for key, value in trainer.callback_metrics.items():
            if "/" not in key:
                continue
            group, metric = key.split("/")

            res[metric] = value
        res["prediction rate (samples/s)"] = round(
            self.data_module.test_n_samples /
            (self.end_test_time - self.start_test_time), 10)
        if hasattr(self, "train_end_time"):
            res["training duration (s)"] = self.train_end_time - self.train_start_time
        with open(json_out, "r") as f:
            r = json.load(f)
            r[self._config.dataset.name] = res
        with open(json_out, "w") as f:
            json.dump(r, f, indent=2)