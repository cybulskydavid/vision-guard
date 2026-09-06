import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset

from src.data.class_subset import ClassSubset


class ClassDataModule(pl.LightningDataModule):
  def __init__(
    self, 
    batch_size: int,
    num_workers: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    train_sampler: Sampler | None = None
  ) -> None:
    super().__init__()

    self.batch_size = batch_size
    self.num_workers = num_workers
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.test_dataset = test_dataset
    self.train_sampler = train_sampler

  def train_dataloader(self) -> DataLoader:
    shuffle = (self.train_sampler is None)
    dataloader = self._get_dataloader(
      self.train_dataset,
      shuffle=shuffle,
      sampler=self.train_sampler)
    return dataloader
  

  def val_dataloader(self) -> DataLoader:
    dataloader = self._get_dataloader(self.val_dataset)
    return dataloader
  

  def test_dataloader(self) -> DataLoader:
    dataloader = self._get_dataloader(self.test_dataset)
    return dataloader
  

  def _get_dataloader(
      self, 
      dataset: Dataset, 
      shuffle = False,
      sampler: Sampler | None = None
    ) -> DataLoader:
    return DataLoader(
      dataset=dataset,
      batch_size=self.batch_size,
      shuffle=shuffle,
      sampler=sampler,
      num_workers=self.num_workers,
      pin_memory=True,
      persistent_workers=True,
      prefetch_factor=4
    )
