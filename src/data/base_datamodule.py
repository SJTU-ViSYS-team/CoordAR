from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import ipdb
import torch
from lightning import LightningDataModule
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    random_split,
    IterableDataset,
)
from tqdm import tqdm

from src.data.collate_importer import CollateImporter


class BaseDataModule(LightningDataModule):
    """`LightningDataModule` for the InstantPose."""

    def __init__(
        self,
        data_train: Dataset,
        data_val: Dataset,
        data_test: Dataset,
        data_predict: Dataset,
        collate_importer: CollateImporter,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `CIRDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        want_to_save = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.save_hyperparameters(want_to_save, logger=False)

        self.data_train: Optional[Dataset] = data_train
        self.data_val: Optional[Dataset] = data_val
        self.data_test: Optional[Dataset] = data_test
        self.data_predict: Optional[Dataset] = data_predict

        self.batch_size_per_device = batch_size

        self.collate_importer = collate_importer

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if isinstance(self.data_train, IterableDataset):
            shuffle = False
        else:
            shuffle = True
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=self.collate_importer.train_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_importer.val_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_importer.test_fn,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_importer.predict_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


def test_dataloader(loader: DataLoader, max_step: int = 10) -> None:
    for i, batch in tqdm(enumerate(loader)):
        if i >= max_step:
            break
    print("Done")


# python -m src.data.cir_datamodule
if __name__ == "__main__":

    dm = BaseDataModule(
        gso_dataset, tless_val_dataset, tless_test_dataset, tless_test_dataset
    )
    train_loader = dm.train_dataloader()
    test_dataloader(train_loader)

    val_loader = dm.val_dataloader()
    test_dataloader(val_loader)

    test_loader = dm.test_dataloader()
    test_dataloader(test_loader)
