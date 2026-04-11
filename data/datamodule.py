"""LightningDataModule for pilot cockpit dataset."""
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
        dc = self.cfg
        img_size = dc.get('img_size', 224)
        sampling = dc.get('sampling', 'uniform')
        segment_window = dc.get('segment_window', 64)
        self.train_dataset = KeyframeClipDataset(
            ann_file=dc.train_ann, data_root=dc.root,
            clip_len=dc.clip_len, frame_interval=dc.frame_interval,
            jitter_range=dc.jitter_range,
            sampling=sampling, segment_window=segment_window,
            is_train=True,
            transform=build_train_transforms(img_size=img_size))
        self.val_dataset = KeyframeClipDataset(
            ann_file=dc.val_ann, data_root=dc.root,
            clip_len=dc.clip_len, frame_interval=dc.frame_interval,
            jitter_range=0,
            sampling=sampling, segment_window=segment_window,
            is_train=False,
            transform=build_val_transforms(img_size=img_size))

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
