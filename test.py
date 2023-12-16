import json
import os
from pathlib import Path
import logging
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch
import torchaudio
from tqdm import tqdm

from hw_as.utils import ROOT_PATH

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


@hydra.main(version_base=None, config_path="hw_as/configs", config_name="test")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    logger = logging.getLogger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = instantiate(config["arch"])

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    logger.info(f"Device {device}")
    model = model.to(device)
    model.eval()

    logger.info("Probabilities:")
    for audio_path in config.audios:
        audio = torchaudio.load(audio_path)[0].to(device).unsqueeze(0)
        logits = model(audio)[0]
        bonafide_prob = torch.softmax(logits, 0)[1]
        logger.info("{} {:.5f}".format(audio_path, bonafide_prob))



if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
