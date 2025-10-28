\
import io, torch
from torch.utils.data import Dataset, DataLoader, random_split

# special tokens
BOS = 256
EOS = 257
PAD = 258
VOCAB_SIZE = 259

def encode_line(s: str):
    b = s.encode("utf-8", errors="ignore")
    return list(b)

def pack_with_specials(ids, max_len, add_bos=True, add_eos=True):
    seq = []
    if add_bos: seq.append(BOS)
    seq.extend(ids[: (max_len - (1 if add_bos else 0) - (1 if add_eos else 0))])
    if add_eos: seq.append(EOS)
    return seq

class ParallelDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str, max_src_len=128, max_tgt_len=128):
        super().__init__()
        with io.open(src_path, "r", encoding="utf-8") as fs, io.open(tgt_path, "r", encoding="utf-8") as ft:
            src_lines = [l.strip() for l in fs if l.strip()]
            tgt_lines = [l.strip() for l in ft if l.strip()]
        assert len(src_lines) == len(tgt_lines), "Parallel files must be line-aligned"
        self.pairs = list(zip(src_lines, tgt_lines))
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        s, t = self.pairs[idx]
        s_ids = encode_line(s); t_ids = encode_line(t)
        src = pack_with_specials(s_ids, self.max_src_len, add_bos=True, add_eos=True)
        tgt_inp = pack_with_specials(t_ids, self.max_tgt_len, add_bos=True, add_eos=False)
        tgt_out = pack_with_specials(t_ids, self.max_tgt_len, add_bos=False, add_eos=True)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt_inp, dtype=torch.long), torch.tensor(tgt_out, dtype=torch.long)

def pad_batch(batch):
    srcs, tins, touts = zip(*batch)
    def pad_to_max(seqs):
        maxlen = max(len(s) for s in seqs)
        out = torch.full((len(seqs), maxlen), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = s
        mask = (out == PAD)
        return out, mask
    src_pad, src_kpm = pad_to_max(srcs)
    tin_pad, tin_kpm = pad_to_max(tins)
    tout_pad, _ = pad_to_max(touts)
    return src_pad, tin_pad, tout_pad, src_kpm, tin_kpm

def get_loaders(src_path, tgt_path, max_src_len=128, max_tgt_len=128, batch_size=64, train_frac=0.95, num_workers=0, seed=42):
    ds = ParallelDataset(src_path, tgt_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    n_train = max(1, int(train_frac * len(ds)))
    n_val = max(1, len(ds) - n_train)
    g = torch.Generator().manual_seed(seed)
    tr, va = random_split(ds, [n_train, n_val], generator=g)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pad_batch, drop_last=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pad_batch, drop_last=False)
    return tr_loader, va_loader
