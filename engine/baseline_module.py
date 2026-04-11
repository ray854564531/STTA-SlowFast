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


class _C3D(nn.Module):
    """VGG-style C3D network (input: B x 3 x 16 x 112 x 112).

    Architecture follows the original Tran et al. 2015 paper:
    5 conv blocks with max-pool, then two FC layers before the classifier.
    Supports pretrained=False (random init) only; pretrained loading is
    left as a future extension since pytorchvideo ≤ 0.1.5 ships no C3D.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv3d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # block 2
            nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # block 3
            nn.Conv3d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # block 4
            nn.Conv3d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # block 5
            nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        # After pooling: T=16→16→8→4→2→1, H/W=112→56→28→14→7→3 (floor)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,T,H,W)
        x = self.features(x)
        return self.classifier(x)


def _remap_c3d_keys(state: dict) -> dict:
    """Remap MMAction2 C3D Sports-1M keys to custom _C3D Sequential keys."""
    mapping = {
        'backbone.conv1a': 'features.0',
        'backbone.conv2a': 'features.3',
        'backbone.conv3a': 'features.6',
        'backbone.conv3b': 'features.8',
        'backbone.conv4a': 'features.11',
        'backbone.conv4b': 'features.13',
        'backbone.conv5a': 'features.16',
        'backbone.conv5b': 'features.18',
        'cls_head.fc1': 'classifier.2',
        'cls_head.fc2': 'classifier.5',
        'cls_head.fc_cls': 'classifier.8',
    }
    new_state = {}
    for k, v in state.items():
        new_key = k
        for old_prefix, new_prefix in mapping.items():
            if k.startswith(old_prefix + '.'):
                new_key = new_prefix + k[len(old_prefix):]
                break
        new_state[new_key] = v
    return new_state


def _build_c3d(num_classes: int, pretrained: bool) -> nn.Module:
    """C3D (VGG-style, input 16×112×112).

    Uses custom _C3D since pytorchvideo ≤ 0.1.5 ships no C3D module.
    When pretrained=True, loads Sports-1M weights from MMAction2 model zoo.
    Backbone weights load cleanly; the classification head is randomly
    initialised since num_classes differs from Sports-1M (487 classes).
    """
    model = _C3D(num_classes=num_classes)
    if pretrained:
        url = ('https://download.openmmlab.com/mmaction/recognition/c3d/'
               'c3d_sports1m_pretrain_20201016-dcc47ddc.pth')
        ckpt = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        state = _remap_c3d_keys(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[C3D] Missing keys ({len(missing)}): {missing[:5]} ...")
    return model


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
