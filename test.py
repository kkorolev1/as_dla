import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import hw_as.model as module_model
from hw_as.trainer import Trainer
from hw_as.utils import ROOT_PATH
from hw_as.utils.parse_config import ConfigParser
from hw_as.text import text_to_sequence

from waveglow import get_wav, get_waveglow

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, test_txt, waveglow_path, output_dir):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

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

    waveglow = get_waveglow(waveglow_path, device=device)

    os.makedirs(os.path.join(output_dir, "texts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)

    with open(test_txt, "r") as f:
        texts = [text.strip() for text in f.readlines()]
    text_cleaners = ["english_cleaners"]
    tokenized_texts = [text_to_sequence(t, text_cleaners) for t in texts]
    sampling_rate = 22050

    for i, (text, tokenized_text) in enumerate(zip(texts, tokenized_texts)):
        with open(os.path.join(output_dir, "texts", f"{i+1}.txt"), "w") as f:
            f.write(text)
        for params in config["parameters_list"]:
            src_seq = torch.tensor(tokenized_text, device=device).unsqueeze(0)
            src_pos = torch.tensor(
                [i + 1 for i in range(len(tokenized_text))], device=device).unsqueeze(0)
            outputs = model(src_seq=src_seq, src_pos=src_pos,
                            alpha=params[0], beta=params[1], gamma=params[2])
            wav = get_wav(outputs["mel_output"].transpose(
                1, 2), waveglow, sampling_rate=sampling_rate).unsqueeze(0)

            torchaudio.save(os.path.join(output_dir, "audio",
                            f"{i+1}_speed={params[0]}_pitch={params[1]}_energy={params[2]}.wav"), wav, sample_rate=sampling_rate)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-t",
        "--test-txt",
        default="test.txt",
        type=str,
        help="Path to test file with 3 sentences",
    )
    args.add_argument(
        "-o",
        "--output-dir",
        default="output",
        type=str,
        help="Output directory",
    )
    args.add_argument(
        "-w",
        "--waveglow-path",
        default="waveglow/pretrained_model/waveglow_256channels.pt",
        type=str,
        help="Path to Waveglow weights",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    # model_config = Path(args.resume).parent / "config_server.json"
    model_config = Path(args.config)
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.test_txt, args.waveglow_path, args.output_dir)
