import zipfile

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Special tokens
BOS = 1
EOS = 2
PAD = 0


def _read_csv_from_zip(zip_path: str, inner_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(inner_name) as f:
            return pd.read_csv(f)


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_ted_frames(zip_path: str = None, transcripts_csv: str = None, meta_csv: str = None):
    """
    Load TED transcripts + metadata either from one ZIP (Kaggle-style: transcripts.csv + ted_main.csv)
    or from separate CSV files.
    """
    if zip_path is not None:
        # try canonical filenames
        try_names = ["transcripts.csv", "ted_main.csv"]
        with zipfile.ZipFile(zip_path, "r") as z:
            names = set(z.namelist())

        # find actual names by case-insensitive match
        def find_name(target):
            for n in names:
                if n.lower().endswith(target):
                    return n
            return None

        trans_name = find_name("transcripts.csv")
        main_name = find_name("ted_main.csv")
        if trans_name is None or main_name is None:
            raise FileNotFoundError("Could not locate transcripts.csv and ted_main.csv inside the ZIP")
        trans_df = _read_csv_from_zip(zip_path, trans_name)
        main_df = _read_csv_from_zip(zip_path, main_name)
        return trans_df, main_df
    else:
        if transcripts_csv is None or meta_csv is None:
            raise ValueError("Provide either zip_path or both transcripts_csv and meta_csv")
        trans_df = _read_csv(transcripts_csv)
        main_df = _read_csv(meta_csv)
        return trans_df, main_df


def _merge_ted(trans_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    # Join on URL; some URLs may have trailing slashes—normalize
    def norm_url(s):
        try:
            s = str(s).strip()
            return s[:-1] if s.endswith("/") else s
        except Exception:
            return s

    trans_df = trans_df.copy()
    main_df = main_df.copy()
    trans_df["url_norm"] = trans_df["url"].map(norm_url)
    main_df["url_norm"] = main_df["url"].map(norm_url)
    df = pd.merge(trans_df, main_df, on="url_norm", suffixes=("_trans", "_meta"))
    # keep essential columns
    keep_cols = ["transcript", "title", "description", "url_norm", "event", "published_date", "views"]
    for c in list(df.columns):
        if c not in keep_cols:
            df.drop(columns=[c], inplace=True, errors="ignore")
    return df


def _maybe_to_ids(text: str):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # byte-level ids
    return list(text.encode("utf-8", errors="ignore"))


def _clip_and_pack(ids, max_len, bos=BOS, eos=EOS, add_bos=True, add_eos=True):
    ids = list(map(int, ids))
    seq = []
    if add_bos: seq.append(bos)
    limit = max_len - (1 if add_bos else 0) - (1 if add_eos else 0)
    if limit > 0:
        seq.extend(ids[:limit])
    if add_eos: seq.append(eos)
    return seq


class TEDSeq2SeqDataset(Dataset):
    def __init__(self, rows, src_field="transcript", tgt_field="title", max_src_len=2048, max_tgt_len=128,
                 min_src_chars=64, bos=BOS, eos=EOS, pad=PAD):
        self.rows = []
        for r in rows:
            s = r.get(src_field, "")
            t = r.get(tgt_field, "")
            if not isinstance(s, str) or not isinstance(t, str):
                s = "" if s is None else str(s)
                t = "" if t is None else str(t)
            if len(s.strip()) < min_src_chars:  # filter too-short transcripts
                continue
            self.rows.append({src_field: s, tgt_field: t})
        self.src_field, self.tgt_field = src_field, tgt_field
        self.max_src_len, self.max_tgt_len = max_src_len, max_tgt_len
        self.bos, self.eos, self.pad = bos, eos, pad

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        s_ids = _maybe_to_ids(r[self.src_field])
        t_ids = _maybe_to_ids(r[self.tgt_field])
        src = _clip_and_pack(s_ids, self.max_src_len, self.bos, self.eos, True, True)
        tgt_inp = _clip_and_pack(t_ids, self.max_tgt_len, self.bos, self.eos, True, False)
        tgt_out = _clip_and_pack(t_ids, self.max_tgt_len, self.bos, self.eos, False, True)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt_inp, dtype=torch.long), torch.tensor(tgt_out,
                                                                                                          dtype=torch.long)


def _pad_batch(seqs, pad_id):
    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs): out[i, :len(s)] = s
    mask = (out == pad_id)
    return out, mask


def ted_collate(batch, pad_id=PAD):
    srcs, tins, touts = zip(*batch)
    src_pad, src_kpm = _pad_batch(srcs, pad_id)
    tin_pad, tin_kpm = _pad_batch(tins, pad_id)
    tout_pad, _ = _pad_batch(touts, pad_id)
    return src_pad, tin_pad, tout_pad, src_kpm, tin_kpm


def get_loaders_from_ted(
        zip_path: str = None,
        transcripts_csv: str = None,
        meta_csv: str = None,
        src_field: str = "transcript",
        tgt_field: str = "title",  # or "description"
        max_src_len: int = 2048,
        max_tgt_len: int = 128,
        min_src_chars: int = 64,
        batch_size: int = 8,  # transcripts are long; keep batch small
        num_workers: int = 0,
        seed: int = 42,
        train_frac: float = 0.96,
        valid_frac: float = 0.02,
        test_frac: float = 0.02,
):
    trans_df, main_df = _load_ted_frames(zip_path=zip_path, transcripts_csv=transcripts_csv, meta_csv=meta_csv)
    df = _merge_ted(trans_df, main_df)

    # rows of dicts
    rows = df.to_dict(orient="records")
    ds = TEDSeq2SeqDataset(rows, src_field=src_field, tgt_field=tgt_field, max_src_len=max_src_len,
                           max_tgt_len=max_tgt_len, min_src_chars=min_src_chars)

    # split
    n = len(ds)
    n_train = int(train_frac * n)
    n_valid = int(valid_frac * n)
    n_test = n - n_train - n_valid
    g = torch.Generator().manual_seed(seed)
    tr, va, te = random_split(ds, [n_train, n_valid, n_test], generator=g)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                           collate_fn=lambda b: ted_collate(b, PAD), drop_last=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           collate_fn=lambda b: ted_collate(b, PAD), drop_last=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           collate_fn=lambda b: ted_collate(b, PAD), drop_last=False)

    return tr_loader, va_loader, te_loader, BOS, EOS, PAD
