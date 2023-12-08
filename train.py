import argparse
import collections
import warnings
import sys
import os

import numpy as np
import torch

import logging

import hydra
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from hw_as.trainer import Trainer
from hw_as.utils import prepare_device
from hw_as.utils.object_loading import get_dataloaders
from hw_as.utils.parse_config import ConfigParser


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(version_base=None, config_path="hw_as/configs", config_name="train")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    config = ConfigParser(OmegaConf.to_container(config))
    
    logger = logging.getLogger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)
    
    # build model architecture, then print to console
    model = instantiate(config["arch"])
    
    logger.info(model)
    logger.info(f'Len epoch {config["trainer"]["len_epoch"]}')
    logger.info(f'Epochs {config["trainer"]["epochs"]}')
    for split in dataloaders:
        logger.info(f'{split} size {len(dataloaders[split].dataset)}')

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"], logger)
    logger.info(f"Device {device}")
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # get function handles of loss and metrics
    loss_module = instantiate(config["loss"]).to(device)
    metrics = [
        instantiate(metric_dict)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config["optimizer"], trainable_params)
    lr_scheduler = instantiate(config["lr_scheduler"], optimizer)
    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()