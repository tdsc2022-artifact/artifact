from os.path import exists, join
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from models.mulvuldeepecker.data_classes import MulVDPSample, MulVDPBatch
from models.mulvuldeepecker.MulVDP_dataset import MulVDPDataset
from utils.vocabulary import Vocabulary_token
from math import ceil


class MulVDPDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary_token):
        super().__init__()
        self._config = config
        self._vocabulary = vocabulary

        self._dataset_dir = join(config.data_folder, config.name,
                                 config.dataset.name)
        self._train_data_file = join(self._dataset_dir, "train.pkl")
        self._val_data_file = join(self._dataset_dir, "val.pkl")
        self._test_data_file = join(self._dataset_dir, "test.pkl")

    def prepare_data(self):
        if not exists(self._dataset_dir):
            raise ValueError(
                f"There is no file in passed path ({self._dataset_dir})")
        # TODO: download data from s3 if not exists

    def setup(self, stage: Optional[str] = None):
        # TODO: collect or convert vocabulary if needed
        pass

    @staticmethod
    def collate_wrapper(batch: List[MulVDPSample]) -> MulVDPBatch:
        return MulVDPBatch(batch)

    def create_dataloader(
        self,
        path: str,
        seq_len: int,
        shuffle: bool,
        batch_size: int,
        n_workers: int,
    ) -> Tuple[DataLoader, int]:
        dataset = MulVDPDataset(path, seq_len, shuffle)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_wrapper,
            num_workers=n_workers,
            pin_memory=True,
        )
        return dataloader, dataset.get_n_samples()

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        dataloader, n_samples = self.create_dataloader(
            self._train_data_file,
            self._config.hyper_parameters.seq_len,
            self._config.hyper_parameters.shuffle_data,
            self._config.hyper_parameters.batch_size,
            self._config.num_workers,
        )
        print(
            f"\napproximate number of steps for train is {ceil(n_samples / self._config.hyper_parameters.batch_size)}"
        )
        return dataloader

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        dataloader, n_samples = self.create_dataloader(
            self._val_data_file,
            self._config.hyper_parameters.seq_len,
            False,
            self._config.hyper_parameters.test_batch_size,
            self._config.num_workers,
        )
        print(
            f"\napproximate number of steps for val is {ceil(n_samples / self._config.hyper_parameters.test_batch_size)}"
        )
        return dataloader

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        dataloader, n_samples = self.create_dataloader(
            self._test_data_file,
            self._config.hyper_parameters.seq_len,
            False,
            self._config.hyper_parameters.test_batch_size,
            self._config.num_workers,
        )
        print(
            f"\napproximate number of steps for test is {ceil(n_samples / self._config.hyper_parameters.test_batch_size)}"
        )
        self.test_n_samples = n_samples
        return dataloader

    def transfer_batch_to_device(self, batch: MulVDPBatch,
                                 device: torch.device) -> MulVDPBatch:
        batch.move_to_device(device)
        return batch
