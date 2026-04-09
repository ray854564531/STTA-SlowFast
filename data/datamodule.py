"""LightningDataModule for pilot cockpit dataset."""
import torch
import pytorch_lightning as pl

_PIN_MEMORY = False
from torch.utils.data import DataLoader
from .dataset import KeyframeClipDataset
from .transforms import build_train_transforms, build_val_transforms


class PilotDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.data

    def setup(self, stage=None):
        self.train_dataset = KeyframeClipDataset(
            ann_file=self.cfg.train_ann, data_root=self.cfg.root,
            clip_len=self.cfg.clip_len, frame_interval=self.cfg.frame_interval,
            jitter_range=self.cfg.jitter_range,
            transform=build_train_transforms())
        self.val_dataset = KeyframeClipDataset(
            ann_file=self.cfg.val_ann, data_root=self.cfg.root,
            clip_len=self.cfg.clip_len, frame_interval=self.cfg.frame_interval,
            jitter_range=0, transform=build_val_transforms())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size,
                          shuffle=True, num_workers=self.cfg.num_workers,
                          pin_memory=_PIN_MEMORY,
                          persistent_workers=(self.cfg.num_workers > 0))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers,
                          pin_memory=_PIN_MEMORY,
                          persistent_workers=(self.cfg.num_workers > 0))
