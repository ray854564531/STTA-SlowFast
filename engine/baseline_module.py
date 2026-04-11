"""Generic LightningModule for baseline video recognition models.

Supports: tsn, tsm, c3d, i3d, r2plus1d.
Model is selected by cfg.model.type and built via model_factory().
Reuses the same optimizer, scheduler, and logging as SlowFastSTTALightningModule.
"""
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from utils.config import Config
from utils.metrics import top_k_accuracy


def model_factory(model_type: str, cfg: Config) -> nn.Module:
    """Build and return a model with pretrained weights loaded.

    Args:
        model_type: One of 'tsn', 'tsm', 'c3d', 'i3d', 'r2plus1d'.
        cfg: Full config object.

    Returns:
        nn.Module with classification head for cfg.data.num_classes.
    """
    mc = cfg.model
    dc = cfg.data
    num_classes = dc.get('num_classes', 8)
    pretrained = mc.get('pretrained', True)
    clip_len = dc.get('clip_len', 8)

    if model_type == 'tsn':
        from models.tsn import TSN
        return TSN(num_classes=num_classes, num_segments=clip_len,
                   pretrained=pretrained)

    if model_type == 'tsm':
        from models.tsm import TSM
        return TSM(num_classes=num_classes, num_segments=clip_len,
                   pretrained=pretrained)

    if model_type == 'c3d':
        return _build_c3d(num_classes=num_classes, pretrained=pretrained)

    if model_type == 'i3d':
        return _build_i3d(num_classes=num_classes, pretrained=pretrained)

    if model_type == 'r2plus1d':
        return _build_r2plus1d(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown model_type: {model_type!r}. "
                     f"Choose from: tsn, tsm, c3d, i3d, r2plus1d")


def _build_c3d(num_classes: int, pretrained: bool) -> nn.Module:
    """C3D with Sports-1M pretrained weights.

    Architecture: pytorchvideo create_c3d (VGG-style 5 conv blocks + FC head).
    Builds with 487 Sports-1M classes, loads weights, then replaces head.
    """
    from pytorchvideo.models.c3d import create_c3d
    model = create_c3d(model_num_class=487)
    if pretrained:
        url = ('https://download.openmmlab.com/mmaction/recognition/c3d/'
               'c3d_sports1m_pretrain_20201016-dcc47ddc.pth')
        ckpt = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        state = _remap_c3d_keys(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[C3D] Missing keys ({len(missing)}): {missing[:5]} ...")
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, num_classes)
    return model


def _remap_c3d_keys(state: dict) -> dict:
    """Remap MMAction2 C3D key names to pytorchvideo C3D key names."""
    mapping = {
        'backbone.conv1a': 'blocks.0.conv',
        'backbone.conv2a': 'blocks.1.conv',
        'backbone.conv3a': 'blocks.2.conv',
        'backbone.conv3b': 'blocks.2.conv',
        'backbone.conv4a': 'blocks.3.conv',
        'backbone.conv4b': 'blocks.3.conv',
        'backbone.conv5a': 'blocks.4.conv',
        'backbone.conv5b': 'blocks.4.conv',
        'cls_head.fc1': 'blocks.5.fc1',
        'cls_head.fc2': 'blocks.5.fc2',
        'cls_head.fc_cls': 'blocks.5.proj',
    }
    new_state = {}
    for k, v in state.items():
        new_key = k
        for old_prefix, new_prefix in mapping.items():
            if k.startswith(old_prefix):
                new_key = k.replace(old_prefix, new_prefix, 1)
                break
        new_state[new_key] = v
    return new_state


def _build_i3d(num_classes: int, pretrained: bool) -> nn.Module:
    """I3D ResNet-50 with Kinetics-400 pretrained weights via pytorchvideo hub."""
    model = torch.hub.load(
        'facebookresearch/pytorchvideo:main', 'i3d_r50',
        pretrained=pretrained)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, num_classes)
    return model


def _build_r2plus1d(num_classes: int, pretrained: bool) -> nn.Module:
    """R(2+1)D ResNet-50 with Kinetics-400 pretrained weights via pytorchvideo hub."""
    model = torch.hub.load(
        'facebookresearch/pytorchvideo:main', 'r2plus1d_r50',
        pretrained=pretrained)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, num_classes)
    return model


class BaselineLightningModule(pl.LightningModule):
    """Training/validation logic for baseline video recognition models.

    Args:
        cfg: Config object with model/train/data/logging sections.
            cfg.model.type must be one of: tsn, tsm, c3d, i3d, r2plus1d.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.save_hyperparameters(cfg.to_dict())
        self.cfg = cfg
        tc = cfg.train
        model_type = cfg.model.get('type', 'tsn')
        self.model = model_factory(model_type, cfg)
        self.criterion = nn.CrossEntropyLoss()
        self._warmup_epochs = tc.warmup_epochs
        self._warmup_start_factor = tc.warmup_start_factor
        self._max_epochs = tc.max_epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc1 = top_k_accuracy(logits, y, k=1)
        acc5 = top_k_accuracy(logits, y, k=min(5, logits.size(1)))
        if self._trainer is not None:
            self.log('train/loss', loss, on_step=True, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            self.log('train/acc1', acc1, on_step=False, on_epoch=True,
                     sync_dist=True)
            self.log('train/acc5', acc5, on_step=False, on_epoch=True,
                     sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc1 = top_k_accuracy(logits, y, k=1)
        acc5 = top_k_accuracy(logits, y, k=min(5, logits.size(1)))
        if self._trainer is not None:
            self.log('val/loss', loss, on_epoch=True, prog_bar=True,
                     sync_dist=True)
            self.log('val/acc1', acc1, on_epoch=True, prog_bar=True,
                     sync_dist=True)
            self.log('val/acc5', acc5, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        tc = self.cfg.train
        optimizer = SGD(
            self.parameters(),
            lr=tc.lr,
            momentum=tc.momentum,
            weight_decay=tc.weight_decay,
        )
        warmup_epochs = self._warmup_epochs
        start_factor = self._warmup_start_factor
        # Cosine period fixed at 160 to match mmaction2 SlowFast config,
        # consistent with SlowFastSTTALightningModule.

        def lr_lambda(epoch: int) -> float:
            cosine = 0.5 * (1.0 + math.cos(math.pi * epoch / 160))
            if epoch < warmup_epochs:
                linear = start_factor + (1.0 - start_factor) * epoch / max(warmup_epochs, 1)
                return linear * cosine
            return cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }

    def on_before_optimizer_step(self, optimizer) -> None:
        nn.utils.clip_grad_norm_(self.parameters(),
                                  max_norm=self.cfg.train.grad_clip)
