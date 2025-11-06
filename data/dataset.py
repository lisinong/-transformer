# dataset_utils.py
import os, glob, json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
BOS = 256
EOS = 257
PAD = 258


@dataclass
class Sample:
    src: str
    tgt: str


def _collect_files(root: str, file_globs: Optional[List[str]]) -> List[str]:
    if os.path.isfile(root):
        return [root]
    patterns = file_globs or ["**/*"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return [f for f in files if os.path.isfile(f)]


def _iter_json_records(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            yield from obj
        elif isinstance(obj, dict):
            yield obj
        return
    except Exception:
        pass
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj
            except Exception:
                continue


class JsonLineSeq2SeqDataset(Dataset):
    def __init__(self, path: str, task: str = "title2text", file_globs: Optional[List[str]] = None):
        self.samples: List[Sample] = []
        files = _collect_files(path, file_globs)
        for fp in files:
            base = os.path.basename(fp)
            if base.startswith("."): continue
            for obj in _iter_json_records(fp):
                try:
                    src, tgt = self._make_pair(obj, task)
                    if src or tgt:
                        self.samples.append(Sample(src, tgt))
                except Exception:
                    continue
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {path}")

    @staticmethod
    def _make_pair(obj: Dict[str, Any], task: str) -> Tuple[str, str]:
        title = (obj.get("title") or "").strip()
        text = (obj.get("text") or "").strip()
        if task == "copy":
            return text, text
        elif task == "title2text":
            return title, text
        else:
            raise ValueError(f"Unknown task {task}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ===== Tokenizer =====
class ByteLevelTokenizer:
    def __init__(self, add_bos=True, add_eos=True, max_len=512):
        self.add_bos = add_bos;
        self.add_eos = add_eos
        self.max_len = max_len
        self.bos_id, self.eos_id, self.pad_id = BOS, EOS, PAD
        self.vocab_size = 259

    def encode(self, s: str) -> List[int]:
        b = s.encode("utf-8", errors="ignore")
        ids = list(b)
        if self.add_bos: ids = [self.bos_id] + ids
        if self.add_eos: ids += [self.eos_id]
        return ids[:self.max_len]
    def decode(self, ids):
        """
        将 token id 序列解码为字符串。
        自动去掉 BOS/EOS/PAD。
        """
        # 过滤特殊符号
        ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        try:
            text = bytes(ids).decode("utf-8", errors="ignore")
        except Exception:
            # 防御性 fallback
            text = "".join(chr(i) for i in ids if 32 <= i < 127)
        return text
    def pad_batch(self, batch: List[List[int]]):
        max_len = max(len(x) for x in batch)
        B = len(batch)
        out = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        mask = torch.ones((B, max_len), dtype=torch.bool)
        for i, seq in enumerate(batch):
            out[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = False
        return out, mask
class SentencePieceTokenizer:
    """
    Wrapper for SentencePiece to match ByteLevelTokenizer's interface.
    Expect an existing .model file; optionally can train one if cfg asks.

    Special IDs follow the sp model:
      - pad_id:   cfg override or sp <pad> if defined, else None
      - bos_id:   sp.bos_id()  (usually 1 or 2 depending on model)
      - eos_id:   sp.eos_id()
    """
    def __init__(self, model_path: str, add_bos: bool = True, add_eos: bool = True,
                 max_len: int = 512, pad_id: int = None):
        if spm is None:
            raise ImportError("sentencepiece is not installed. Please `pip install sentencepiece`.")
        self.sp = spm.SentencePieceProcessor()
        ok = self.sp.Load(model_path)
        if not ok:
            raise RuntimeError(f"Failed to load SentencePiece model: {model_path}")

        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_len = max_len

        # pull ids from the model; allow pad override from cfg
        self.pad_id = pad_id if pad_id is not None else (self.sp.pad_id() if self.sp.pad_id() >= 0 else None)
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else None
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else None

        self.vocab_size = int(self.sp.GetPieceSize())

    # 与 ByteLevelTokenizer 一致的 encode/decode 接口
    def encode(self, s: str):
        ids = self.sp.EncodeAsIds(s)
        if self.add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids[:self.max_len]

    def decode(self, ids):
        # 去掉特殊符号 & 在 EOS 截断
        specials = set(x for x in (self.pad_id, self.bos_id, self.eos_id) if x is not None)
        clean = []
        for t in ids:
            if self.eos_id is not None and t == self.eos_id:
                break
            if t not in specials:
                clean.append(t)
        try:
            return self.sp.DecodeIds(clean)
        except Exception:
            # 极端情况下仍给出可读回退
            return "".join(str(t) for t in clean)

    def pad_batch(self, batch):
        max_len = max(len(x) for x in batch)
        B = len(batch)
        pad_val = self.pad_id if self.pad_id is not None else 0
        out = torch.full((B, max_len), pad_val, dtype=torch.long)
        mask = torch.ones((B, max_len), dtype=torch.bool)
        for i, seq in enumerate(batch):
            out[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = False
        return out, mask

class Seq2SeqCollator:
    def __init__(self, tok: ByteLevelTokenizer, max_src_len, max_tgt_len):
        self.tok = tok
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __call__(self, batch: List[Sample]):
        src_ids = [self.tok.encode(x.src)[:self.max_src_len] for x in batch]
        tgt_ids = [self.tok.encode(x.tgt)[:self.max_tgt_len] for x in batch]
        src, src_mask = self.tok.pad_batch(src_ids)
        tgt, tgt_mask = self.tok.pad_batch(tgt_ids)
        dec_in = tgt[:, :-1]
        dec_mask = tgt_mask[:, :-1]
        target = tgt[:, 1:]
        return dict(src=src, src_padmask=src_mask, dec_in=dec_in, dec_padmask=dec_mask, target=target)

def ensure_sentencepiece_model(cfg):
    """
    若 data.tokenizer.type == 'spm' 且 spm_model 不存在：
    1) 用原数据读取方式构造训练语料（直接遍历 JsonLineSeq2SeqDataset(train_path)）
    2) 训练 SentencePiece 模型（unigram）
    """
    tok_cfg = (cfg.get("data") or {}).get("tokenizer", {})
    if tok_cfg.get("type", "byte") != "spm":
        return

    if spm is None:
        raise ImportError("Please `pip install sentencepiece` to use subword tokenizer.")

    spm_model = tok_cfg.get("spm_model", "./spm/spm.model")
    spm_dir = os.path.dirname(spm_model) or "."
    spm_prefix = os.path.splitext(spm_model)[0]
    vocab_size = int(tok_cfg.get("spm_vocab_size", 8000))
    sample_size = int(tok_cfg.get("spm_sample_size", 0))  # 0=全量
    task = (cfg.get("data") or {}).get("task", "title2text")
    train_path = (cfg.get("data") or {}).get("train_path")
    file_globs = (cfg.get("data") or {}).get("file_globs")

    if os.path.exists(spm_model):
        print(f"[SPM] Found existing SentencePiece model: {spm_model}")
        return

    os.makedirs(spm_dir, exist_ok=True)
    corpus_file = os.path.join(spm_dir, "spm_corpus.txt")
    print(f"[SPM] Building corpus with original dataset reader: {train_path} (task={task})")

    # —— 用“原来的数据处理方法”构造语料 —— #
    dataset = JsonLineSeq2SeqDataset(train_path, task=task, file_globs=file_globs)
    written = 0
    with open(corpus_file, "w", encoding="utf-8") as out:
        for smp in dataset:
            if smp.src:
                out.write(smp.src.replace("\n", " ") + "\n")
                written += 1
            if smp.tgt:
                out.write(smp.tgt.replace("\n", " ") + "\n")
                written += 1
            if sample_size > 0 and written >= sample_size:
                break

    if written == 0:
        raise RuntimeError("[SPM] Corpus empty. Please check your train_path/task/fields.")

    print(f"[SPM] Corpus ready: {corpus_file} (lines: {written})")
    print(f"[SPM] Training SPM: vocab_size={vocab_size}, model_type=unigram")

    spm.SentencePieceTrainer.Train(
        input=corpus_file,
        model_prefix=spm_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,   # 中文推荐
        model_type="unigram",
        bos_id=1, eos_id=2, pad_id=0, unk_id=3,
        num_threads=max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))
    )
    print(f"[SPM] Model saved: {spm_model}")

def build_dataloaders(cfg):
    data_cfg = cfg["data"]
    tok_cfg = data_cfg.get("tokenizer", {})
    if tok_cfg.get("type", "byte") == "spm":
        ensure_sentencepiece_model(cfg)
    train_set = JsonLineSeq2SeqDataset(data_cfg["train_path"], task=data_cfg["task"],
                                       file_globs=data_cfg.get("file_globs"))
    val_set = JsonLineSeq2SeqDataset(data_cfg["val_path"], task=data_cfg["task"], file_globs=data_cfg.get("file_globs"))
    test_set = JsonLineSeq2SeqDataset(data_cfg["test_path"], task=data_cfg["task"],
                                      file_globs=data_cfg.get("file_globs"))
    max_len = max(data_cfg["max_src_len"], data_cfg["max_tgt_len"])
     # === 选择分词器：byte-level 或 sentencepiece ===
    tok_cfg = data_cfg.get("tokenizer", {})  # 新增：从配置读取
    tok_type = tok_cfg.get("type", "byte")   # "byte" | "spm"
    if tok_type == "spm":
        model_path = tok_cfg.get("spm_model","./spm/spm.model")
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("tokenizer.type='spm' but data.tokenizer.spm_model not found.")
        tok = SentencePieceTokenizer(
            model_path=model_path,
            add_bos=tok_cfg.get("add_bos", True),
            add_eos=tok_cfg.get("add_eos", True),
            max_len=max_len,
            pad_id=tok_cfg.get("pad_id", None)
        )
    else:
        # 默认仍支持你原来的字节级分词
        tok = ByteLevelTokenizer(
            add_bos=tok_cfg.get("add_bos", True),
            add_eos=tok_cfg.get("add_eos", True),
            max_len=max_len
        )
    collate = Seq2SeqCollator(tok, data_cfg["max_src_len"], data_cfg["max_tgt_len"])
    train_loader = DataLoader(train_set, batch_size=data_cfg["batch_size"], shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_set,   batch_size=data_cfg["eval_batch_size"], shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_set,  batch_size=data_cfg["eval_batch_size"], shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader,tok
