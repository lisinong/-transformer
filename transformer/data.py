import io
import torch

from torch.utils.data import Dataset, DataLoader, random_split


class ByteDataset(Dataset):
    """
    Simple byte-level character dataset for next-token prediction.
    Reads a text file (utf-8), encodes to bytes in [0,255].
    """

    def __init__(self, path: str, block_size: int = 256):
        super().__init__()
        with io.open(path, "r", encoding="utf-8") as f:
            text = f.read()
        data = text.encode("utf-8", errors="ignore")
        self.data = torch.tensor(list(data), dtype=torch.long)
        self.block_size = block_size
        if len(self.data) <= block_size + 1:
            # pad by repeating to ensure enough length
            reps = (block_size + 2) // max(1, len(self.data)) + 1
            self.data = self.data.repeat(reps)

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y


def get_loaders(path, block_size=256, batch_size=64, train_frac=0.9, num_workers=0, seed=42):
    ds = ByteDataset(path, block_size=block_size)
    n_train = int(train_frac * len(ds))
    n_val = len(ds) - n_train
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, val_loader
