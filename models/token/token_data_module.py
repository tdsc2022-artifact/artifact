from os.path import exists, join
from typing import List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from models.token.data_classes import TokensSample, TokensBatch
from models.token.token_dataset import TokenDataset
from utils.vocabulary import Vocabulary_token


class TokenDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary_token):
        super().__init__()
        self._config = config
        self._vocabulary = vocabulary

        self._dataset_dir = join(config.data_folder, config.name, config.dataset.name)
        self._train_data_file = join(self._dataset_dir,
                                     "train.txt")
        self._val_data_file = join(self._dataset_dir,
                                   "val.txt")
        self._test_data_file = join(self._dataset_dir,
                                    "test.txt")

    def prepare_data(self):
        if not exists(self._dataset_dir):
            raise ValueError(
                f"There is no file in passed path ({self._dataset_dir})")
        # TODO: download data from s3 if not exists

    def setup(self, stage: Optional[str] = None):
        # TODO: collect or convert vocabulary if needed
        pass

    @staticmethod
    def collate_wrapper(batch: List[TokensSample]) -> TokensBatch:
        return TokensBatch(batch)

    def _create_dataset(self, data_file: str) -> Dataset:
        return TokenDataset(
            data_file,
            self._config,
            self._vocabulary,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = self._create_dataset(
            self._train_data_file)
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.batch_size,
            shuffle=self._config.hyper_parameters.shuffle_data,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = self._create_dataset(self._val_data_file)
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.test_batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = self._create_dataset(self._test_data_file)
        self.test_n_samples = dataset.get_n_samples()
        return DataLoader(
            dataset,
            batch_size=self._config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def transfer_batch_to_device(self, batch: TokensBatch,
                                 device: torch.device) -> TokensBatch:
        batch.move_to_device(device)
        return batch
