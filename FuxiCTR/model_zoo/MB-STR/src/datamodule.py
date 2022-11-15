
import pytorch_lightning as pl
from typing import List, Union
from .datasets import dataset_factory
from .dataloaders import RecDataloader

class RecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_code: str = None,
        target_behavior: str = None,
        multi_behavior: Union[bool, List] = None,
        min_uc: int = None,
        num_items: int = None,
        max_len: int = None,
        mask_prob: float = None,
        num_workers: int = None,
        val_negative_sampler_code: str = None,
        val_negative_sample_size: int = None,
        train_batch_size: int = None,
        val_batch_size: int = None,
        predict_only_target: bool = None,
    ):
        super().__init__()
        self.dataset_code = dataset_code
        self.min_uc = min_uc
        self.target_behavior = target_behavior
        self.multi_behavior = multi_behavior
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.num_workers = num_workers
        self.val_negative_sampler_code = val_negative_sampler_code
        self.val_negative_sample_size = val_negative_sample_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.predict_only_target = predict_only_target

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        dataset_factory(
            self.dataset_code,
            self.target_behavior,
            self.multi_behavior,
            self.min_uc,
            )

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.dataset = dataset_factory(
            self.dataset_code,
            self.target_behavior,
            self.multi_behavior,
            self.min_uc,
            )
    
        self.dataloader = RecDataloader(
            self.dataset,
            self.max_len,
            self.mask_prob,
            self.num_items,
            self.num_workers,
            self.val_negative_sampler_code,
            self.val_negative_sample_size,
            self.train_batch_size,
            self.val_batch_size,
            self.predict_only_target,
        )

    def train_dataloader(self):
        return self.dataloader.get_train_loader()
    def val_dataloader(self):
        return self.dataloader.get_val_loader()

class RecDataModuleNeg(pl.LightningDataModule):
    def __init__(
        self,
        dataset_code: str = None,
        target_behavior: str = None,
        multi_behavior: bool = None,
        min_uc: int = None,
        num_items: int = None,
        max_len: int = None,
        mask_prob: float = None,
        num_workers: int = None,
        train_negative_sampler_code: str = None,
        train_negative_sample_size: int = None,
        val_negative_sampler_code: str = None,
        val_negative_sample_size: int = None,
        train_batch_size: int = None,
        val_batch_size: int = None,
        predict_only_target: bool = None,
    ):
        super().__init__()
        self.dataset_code = dataset_code
        self.min_uc = min_uc
        self.target_behavior = target_behavior
        self.multi_behavior = multi_behavior
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.num_workers = num_workers
        self.train_negative_sampler_code = train_negative_sampler_code
        self.train_negative_sample_size = train_negative_sample_size
        self.val_negative_sampler_code = val_negative_sampler_code
        self.val_negative_sample_size = val_negative_sample_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.predict_only_target = predict_only_target

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        dataset_factory(
            self.dataset_code,
            self.target_behavior,
            self.multi_behavior,
            self.min_uc,
            )

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.dataset = dataset_factory(
            self.dataset_code,
            self.target_behavior,
            self.multi_behavior,
            self.min_uc,
            )
    
        self.dataloader = RecDataloaderNeg(
            self.dataset,
            self.max_len,
            self.mask_prob,
            self.num_items,
            self.num_workers,
            self.train_negative_sampler_code,
            self.train_negative_sample_size,
            self.val_negative_sampler_code,
            self.val_negative_sample_size,
            self.train_batch_size,
            self.val_batch_size,
            self.predict_only_target,
        )

    def train_dataloader(self):
        return self.dataloader.get_train_loader()
    def val_dataloader(self):
        return self.dataloader.get_val_loader()