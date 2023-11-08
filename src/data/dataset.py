import torch
import librosa
import numpy as np
from typing import Literal
from datasets import load_dataset, Audio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src.data.vocab import SimpleVocab

MAX_FRAMES = 200

class CommonVoice(Dataset):
    def __init__(
        self, split: Literal["train", "validation", "test"], streaming: bool = False
    ):
        self.dataset = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            "en",
            split=split,
            streaming=streaming,
        )

        #self.dataset = self.dataset.select(range(100))
        self._preprocess()
        self.vocab = SimpleVocab(self.dataset, max_size=10000)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        x = torch.tensor(item["audio"]["array"], dtype=torch.float32)
        x = librosa.feature.mfcc(y=x.numpy(), sr=16_000, n_mfcc=13)
        y = self.vocab.encode(item["sentence"])

        # Pad the MFCCs if necessary
        pad_width = MAX_FRAMES - x.shape[1]
        if pad_width > 0:
            x = np.pad(x, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif pad_width < 0:
            x = x[:, :MAX_FRAMES]

        return x, torch.tensor(y, dtype=torch.long)

    def _preprocess(self):
        feature_to_remove = [
            "client_id",
            "path",
            "up_votes",
            "down_votes",
            "age",
            "gender",
            "accent",
            "locale",
            "segment",
        ]
        self.dataset = self.dataset.remove_columns(feature_to_remove)
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16_000))