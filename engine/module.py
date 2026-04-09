"""PyTorch Lightning training module for SlowFast + STTA."""
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from models.slowfast_stta import SlowFastWithSTTA
from models.head import SlowFastHead
from utils.config import Config
from utils.metrics import top_k_accuracy


class SlowFastSTTALightningModule(pl.LightningModule):
    """Training/validation logic for SlowFast + STTripletAttention.

    Args:
        cfg: Config object with model/train/data/logging sections.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        tc = cfg.train

        self.backbone = SlowFastWithSTTA(
            resample_rate=mc.resample_rate,
            speed_ratio=mc.speed_ratio,
            channel_ratio=mc.channel_ratio,
            slow_pathway=dict(
                depth=50,
                lateral=True,
                conv1_kernel=(1, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(0, 0, 1, 1),
                spatial_strides=(1, 2, 2, 1),
                speed_ratio=mc.speed_ratio,
                channel_ratio=mc.channel_ratio,
            ),
            fast_pathway=dict(
                depth=50,
                lateral=False,
                base_channels=8,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(1, 1, 1, 1),
                spatial_strides=(1, 2, 2, 1),
            ),
            stta_kernel_size=mc.kernel_size,
            stta_stages=mc.stta_stages,
            enable_stta=mc.get('enable_stta', True),
            stta_enable_tcw=mc.get('enable_tcw', True),
            stta_enable_tch=mc.get('enable_tch', True),
            stta_enable_thw=mc.get('enable_thw', True),
        )

        num_classes = cfg.data.get('num_classes', 8)
        self.head = SlowFastHead(
            in_channels=2304,
            num_classes=num_classes,
            dropout=mc.dropout,
        )
        self.criterion = nn.CrossEntropyLoss()
        self._warmup_epochs = tc.warmup_epochs
        self._warmup_start_factor = tc.warmup_start_factor
        self._max_epochs = tc.max_epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc1 = top_k_accuracy(logits, y, k=1)
        acc5 = top_k_accuracy(logits, y, k=min(5, logits.size(1)))
        if self._trainer is not None:
            self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train/acc1', acc1, on_step=False, on_epoch=True)
            self.log('train/acc5', acc5, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc1 = top_k_accuracy(logits, y, k=1)
        acc5 = top_k_accuracy(logits, y, k=min(5, logits.size(1)))
        if self._trainer is not None:
            self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val/acc1', acc1, on_epoch=True, prog_bar=True, sync_dist=True)
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
        total_epochs = self._max_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return start_factor + (1.0 - start_factor) * epoch / max(warmup_epochs, 1)
            t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }

    def on_before_optimizer_step(self, optimizer) -> None:
        """Gradient clipping (max_norm=40, matches mmaction2 config)."""
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.cfg.train.grad_clip)
