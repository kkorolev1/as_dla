import numpy as np
import torch
import torchaudio
from pathlib import Path

def copy_or_trim(audio, num_samples):
    if audio.shape[-1] < num_samples:
        audio = audio.repeat(1, num_samples // audio.shape[-1] + 1)
    audio = audio[..., :num_samples]
    return audio

class ASVspoofDataset(torch.utils.data.Dataset):
    def __init__(self, protocol_path, audio_dir, num_samples, limit=None):
        self.protocol_path = Path(protocol_path)
        self.audio_dir = Path(audio_dir)
        self.num_samples = num_samples
        self.data = []
        
        with open(protocol_path, "r") as f:
            for line in f.readlines():
                items = line.strip().split(" ")
                utterance_id = items[1]
                target = 1 if items[-1] == "bonafide" else 0
                self.data.append({
                    "audio_path": self.audio_dir / f"{utterance_id}.flac",
                    "target": target
                })
        
        if limit is not None:
            np.random.seed(42)
            np.random.shuffle(self.data)
            self.data = self.data[:limit]
        
    def __getitem__(self, index):
        item_dict = self.data[index]
        item_dict.update({
            "audio": torchaudio.load(item_dict["audio_path"])[0]
        })
        return item_dict
    
    def __len__(self):
        return len(self.data)
    
    def get_collate(self):
        def collate(batch):
            output = {}
            output["audio"] = torch.cat([copy_or_trim(item["audio"], self.num_samples) for item in batch]).unsqueeze(1)
            output["audio_path"] = [item["audio_path"] for item in batch]
            output["target"] = torch.tensor([item["target"] for item in batch], dtype=torch.long)
            return output
        return collate