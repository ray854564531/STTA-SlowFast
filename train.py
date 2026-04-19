"""Training entry point.

Usage:
    uv run train.py --config configs/slowfast_stta_full.yaml
"""
import argparse
import os
import sys

import torch
import yaml
import pytorch_lightning as pl

torch.set_float32_matmul_precision('high')
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data.datamodule import PilotDataModule
from engine.module import SlowFastSTTALightningModule
from engine.baseline_module import BaselineLightningModule
from utils.config import load_config

_BASELINE_TYPES = {'tsn', 'tsm', 'c3d', 'i3d', 'r2plus1d'}


def parse_args():
    parser = argparse.ArgumentParser(description='Train SlowFast + STTA')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    return parser.parse_args()


def build_loggers(cfg, config_path: str):
    loggers = []
    lc = cfg.get('logging', None)
    if lc is None:
        return loggers, None

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    wandb_run_name = lc.get('wandb_name', None) or config_name

    wandb_project = lc.get('wandb_project', 'pilot-action-recognition')
    wandb_mode = lc.get('wandb_mode', 'online')
    tb_logger = TensorBoardLogger(save_dir='logs/tb', name=config_name)
    if wandb_mode != 'disabled':
        loggers.append(WandbLogger(project=wandb_project, name=wandb_run_name, mode=wandb_mode))
    loggers.append(tb_logger)
    return loggers, tb_logger


def main():
    args = parse_args()
    cfg = load_config(args.config)
    pl.seed_everything(cfg.train.get('seed', 42))

    model_type = cfg.model.get('type', None)
    if model_type in _BASELINE_TYPES:
        module = BaselineLightningModule(cfg)
    else:
        module = SlowFastSTTALightningModule(cfg)
    datamodule = PilotDataModule(cfg)

    loggers, tb_logger = build_loggers(cfg, args.config)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    version = f'version_{tb_logger.version}' if tb_logger is not None else 'version_0'
    ckpt_dir = os.path.join('work_dirs', config_name, version)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, allow_unicode=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val/acc1',
        mode='max',
        save_top_k=cfg.logging.get('save_top_k', 3),
        filename='epoch={epoch:03d}-val_acc1={val/acc1:.4f}',
        auto_insert_metric_name=False,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

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
        precision='32',
        gradient_clip_val=None,  # Handled in on_before_optimizer_step
        devices=devices,
        strategy=strategy,
    )

    trainer.fit(module, datamodule)


if __name__ == '__main__':
    main()
