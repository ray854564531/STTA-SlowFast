import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import PIL.Image

from engine.module import SlowFastSTTALightningModule
from utils.config import Config


def _make_cfg(enable_tcw=True, enable_tch=True, enable_thw=True):
    return Config({
        'model': {
            'resample_rate': 4,
            'speed_ratio': 4,
            'channel_ratio': 8,
            'kernel_size': 7,
            'dropout': 0.5,
            'stta_stages': [True, True, True, True],
            'enable_tcw': enable_tcw,
            'enable_tch': enable_tch,
            'enable_thw': enable_thw,
        },
        'data': {'num_classes': 8},
        'train': {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'grad_clip': 40,
            'max_epochs': 150,
            'warmup_epochs': 5,
            'warmup_start_factor': 0.1,
        },
        'logging': {},
    })


def test_module_forward():
    """单个 batch 前向传播正常。"""
    cfg = _make_cfg()
    module = SlowFastSTTALightningModule(cfg)
    x = torch.randn(2, 3, 32, 224, 224)
    logits = module(x)
    assert logits.shape == (2, 8)


def test_training_step():
    cfg = _make_cfg()
    module = SlowFastSTTALightningModule(cfg)
    x = torch.randn(2, 3, 32, 224, 224)
    y = torch.tensor([0, 3])
    loss = module.training_step((x, y), batch_idx=0)
    assert loss.item() > 0


def test_ablation_single_branch():
    cfg = _make_cfg(enable_tcw=True, enable_tch=False, enable_thw=False)
    module = SlowFastSTTALightningModule(cfg)
    x = torch.randn(1, 3, 32, 224, 224)
    logits = module(x)
    assert logits.shape == (1, 8)


def test_end_to_end_with_real_data(tmp_path):
    """用假数据跑完整 train_step：数据加载 -> 模型 -> loss."""
    from data.datamodule import PilotDataModule

    # 创建假数据
    frames_dir = tmp_path / "frames"
    for vid_id in [1]:
        vid_dir = frames_dir / str(vid_id)
        vid_dir.mkdir(parents=True)
        for i in range(1, 101):
            img = PIL.Image.new('RGB', (256, 256), color=(100, 150, 200))
            img.save(vid_dir / f"img_{i:05d}.jpg")

    ann_train = tmp_path / "train.csv"
    ann_train.write_text("video_id,keyframe_id,action_id\n1,50,1\n1,60,2\n")
    ann_val = tmp_path / "val.csv"
    ann_val.write_text("video_id,keyframe_id,action_id\n1,70,3\n")

    cfg = Config({
        'model': {
            'resample_rate': 4, 'speed_ratio': 4, 'channel_ratio': 8,
            'kernel_size': 7, 'dropout': 0.5,
            'stta_stages': [True, True, True, True],
            'enable_tcw': True, 'enable_tch': True, 'enable_thw': True,
        },
        'data': {
            'root': str(frames_dir),
            'train_ann': str(ann_train),
            'val_ann': str(ann_val),
            'num_classes': 8, 'clip_len': 32, 'frame_interval': 1,
            'jitter_range': 2, 'batch_size': 2, 'num_workers': 0,
        },
        'train': {
            'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4,
            'grad_clip': 40, 'max_epochs': 150,
            'warmup_epochs': 5, 'warmup_start_factor': 0.1,
        },
        'logging': {},
    })

    module = SlowFastSTTALightningModule(cfg)
    dm = PilotDataModule(cfg)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    loss = module.training_step(batch, batch_idx=0)
    assert loss.item() > 0
    print(f"End-to-end loss: {loss.item():.4f}")
