import torch
from typing import Literal
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        x = torch.tensor(item["audio"]["array"])
        y = item["sentence"]
        return x, y


def collate_batch(batch):
    batch_x = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
    batch_y = [item[1] for item in batch]
    batch_x_padded = pad_sequence(batch_x, batch_first=True, padding_value=0)
    return batch_x_padded, batch_y
