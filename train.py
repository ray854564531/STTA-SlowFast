"""Training entry point.

Usage:
    uv run train.py --config configs/slowfast_stta_full.yaml
"""
import argparse
import os
import sys

import torch
import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data.datamodule import PilotDataModule
from engine.module import SlowFastSTTALightningModule
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train SlowFast + STTA')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    return parser.parse_args()


def build_loggers(cfg, config_path: str):
    loggers = []
    lc = cfg.get('logging', None)
    if lc is None:
        return loggers

    run_name = lc.get('wandb_name', None) or os.path.splitext(
        os.path.basename(config_path))[0]

    wandb_project = lc.get('wandb_project', 'pilot-action-recognition')
    loggers.append(WandbLogger(project=wandb_project, name=run_name))
    loggers.append(TensorBoardLogger(save_dir='logs/tb', name=run_name))
    return loggers


def main():
    args = parse_args()
    cfg = load_config(args.config)
    pl.seed_everything(cfg.train.get('seed', 42))

    module = SlowFastSTTALightningModule(cfg)
    datamodule = PilotDataModule(cfg)

    checkpoint_cb = ModelCheckpoint(
        dirpath=f'work_dirs/{os.path.splitext(os.path.basename(args.config))[0]}',
        monitor='val/acc1',
        mode='max',
        save_top_k=cfg.logging.get('save_top_k', 3),
        filename='epoch{epoch:03d}-acc{val_acc1:.4f}',
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    loggers = build_loggers(cfg, args.config)

    val_check = cfg.logging.get('val_check_interval', 2)
    trainer_cfg = cfg.get('trainer', None)
    devices = trainer_cfg.get('devices', 1) if trainer_cfg else 1
    strategy = trainer_cfg.get('strategy', 'auto') if trainer_cfg else 'auto'
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        check_val_every_n_epoch=val_check,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=loggers if loggers else True,
        log_every_n_steps=cfg.logging.get('log_every_n_steps', 50),
        precision='16-mixed',
        gradient_clip_val=None,  # Handled in on_before_optimizer_step
        devices=devices,
        strategy=strategy,
    )

    trainer.fit(module, datamodule)


if __name__ == '__main__':
    main()
