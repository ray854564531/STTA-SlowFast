"""LightningDataModule wrapping KineticsVideoDataset."""
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .kinetics_dataset import KineticsVideoDataset
from .video_transforms import (
    build_train_video_transform,
    build_val_video_transform,
    build_test_three_crop_transform,
)


class KineticsDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        dc = self.cfg.data
        mean = list(dc.get('mean', [0.45, 0.45, 0.45]))
        std = list(dc.get('std', [0.225, 0.225, 0.225]))
        img_size = dc.get('img_size', 224)
        short_train = list(dc.get('short_side_scale_train', [256, 320]))
        short_eval = dc.get('short_side_scale_eval', 256)

        train_tfm = build_train_video_transform(short_train, img_size, mean, std)
        val_tfm = build_val_video_transform(short_eval, img_size, mean, std)

        common = dict(
            root=dc.root,
            clip_len=dc.clip_len,
            frame_interval=dc.frame_interval,
            decode_backend=dc.get('decode_backend', 'decord'),
        )
        self.train_dataset = KineticsVideoDataset(
            split='train', mode='train', transform=train_tfm, **common)
        self.val_dataset = KineticsVideoDataset(
            split='val', mode='val', transform=val_tfm, **common)

        test_cfg = self.cfg.get('test', {}) or {}
        num_clips = test_cfg.get('num_clips', 1)
        num_crops = test_cfg.get('num_crops', 1)
        if num_crops == 3:
            test_tfm = build_test_three_crop_transform(
                short_eval, img_size, mean, std)
        else:
            test_tfm = val_tfm
        self.test_dataset = KineticsVideoDataset(
            split='val', mode='test',
            num_clips=num_clips, num_crops=num_crops,
            transform=test_tfm, **common)

    def train_dataloader(self):
        dc = self.cfg.data
        return DataLoader(
            self.train_dataset, batch_size=dc.batch_size,
            shuffle=True, num_workers=dc.num_workers,
            pin_memory=False,
            persistent_workers=(dc.num_workers > 0),
            drop_last=True,
        )

    def val_dataloader(self):
        dc = self.cfg.data
        return DataLoader(
            self.val_dataset, batch_size=dc.batch_size,
            shuffle=False, num_workers=dc.num_workers,
            pin_memory=False,
            persistent_workers=(dc.num_workers > 0),
        )

    def test_dataloader(self):
        dc = self.cfg.data
        tc = self.cfg.get('test', {}) or {}
        bs = tc.get('batch_size', 1)
        return DataLoader(
            self.test_dataset, batch_size=bs,
            shuffle=False, num_workers=dc.num_workers,
            pin_memory=False,
            persistent_workers=(dc.num_workers > 0),
        )
