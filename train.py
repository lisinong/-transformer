# train_utils.py
import argparse
import json
import os
import time
import torch

import matplotlib
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformer.model_seq2seq import TransformerSeq2Seq, PAD
from dataset.dataset import build_dataloaders
import math
from collections import Counter
import time
import sentencepiece as spm


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss doesn't improve.

    Args:
        patience (int): How many epochs to wait after last improvement.
        delta (float): Minimum change to qualify as an improvement.
        verbose (bool): Print messages when triggered.
    """

    def __init__(self, patience=5, delta=0.0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def step(self, val_loss, epoch):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] No improvement for {self.patience} epochs. "
                          f"Best epoch: {self.best_epoch}, best val_loss={self.best_loss:.4f}")


def iter_with_pbar(it, total=None, desc="", leave=False):
    if tqdm is None:
        return it  # 没装tqdm时退化为原迭代器
    return tqdm(it, total=total, desc=desc, leave=leave, dynamic_ncols=True)


def accuracy_from_logits(logits, target, ignore_index=PAD):
    pred = logits.argmax(dim=-1)
    mask = target.ne(ignore_index)
    correct = (pred.eq(target) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def plot_training_curves(hist, output_dir):
    """
    绘制并保存训练/验证的损失和精度曲线。

    参数:
        hist: dict，包含以下键：
            - train_loss
            - val_loss
            - train_acc
            - val_acc
        output_dir: str，输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        # --- 损失曲线 ---
        plt.figure()
        plt.plot(hist["train_loss"], label="train_loss")
        plt.plot(hist["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()

        # --- 精度曲线 ---
        plt.figure()
        plt.plot([a * 100 for a in hist["train_acc"]], label="train_acc(%)")
        plt.plot([a * 100 for a in hist["val_acc"]], label="val_acc(%)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "acc_curve.png"))
        plt.close()

        print(f"[Info] Saved training curves to {output_dir}")
    except Exception as e:
        print(f"[Warn] plot_training_curves failed: {e}")


def evaluate(model, loader, criterion, device, show_pbar=False):
    model.eval()
    total_loss = total_acc = 0
    steps = 0
    it = iter_with_pbar(loader, total=len(loader), desc="Valid", leave=False) if show_pbar else loader
    with torch.no_grad():
        for batch in it:
            src, dec_in, target = batch["src"].to(device), batch["dec_in"].to(device), batch["target"].to(device)
            src_pad, dec_pad = batch["src_padmask"].to(device), batch["dec_padmask"].to(device)
            logits = model(src, dec_in, src_key_padding_mask=src_pad, tgt_key_padding_mask=dec_pad)
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            acc = accuracy_from_logits(logits, target)
            total_loss += loss.item();
            total_acc += acc;
            steps += 1
            if show_pbar and tqdm is not None:
                it.set_postfix(loss=f"{total_loss / steps:.3f}", acc=f"{(total_acc / steps) * 100:.2f}%")
    return dict(loss=total_loss / steps, acc=total_acc / steps)


def train(cfg):
    import json
    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, tok = build_dataloaders(cfg)

    mcfg = cfg["model"]
    model = TransformerSeq2Seq(
        vocab_size=tok.vocab_size, d_model=mcfg["d_model"], n_heads=mcfg["n_heads"],
        d_ff=mcfg["d_ff"], n_layers=mcfg["n_layers"], dropout=mcfg["dropout"],
        use_bias=True, positional_encoding=mcfg["positional_encoding"],
        max_seq_len=max(cfg["data"]["max_src_len"], cfg["data"]["max_tgt_len"]),
        tie_weights=mcfg["tie_weights"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    # === 添加 EarlyStopping ===
    patience = cfg["training"].get("early_stop_patience", 5)
    early_stopper = EarlyStopping(patience=patience, delta=1e-3, verbose=True)
    best = float("inf")
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    out_dir = cfg["training"]["output_dir"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        tl = ta = 0.0
        steps = 0
        t0 = time.time()
        pbar = iter_with_pbar(train_loader, total=len(train_loader), desc=f"Train {epoch}/{cfg['training']['epochs']}",
                              leave=False)
        for batch in pbar:
            src, dec_in, target = batch["src"].to(device), batch["dec_in"].to(device), batch["target"].to(device)
            src_pad, dec_pad = batch["src_padmask"].to(device), batch["dec_padmask"].to(device)

            optimizer.zero_grad()
            logits = model(src, dec_in, src_key_padding_mask=src_pad, tgt_key_padding_mask=dec_pad)
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()

            acc = accuracy_from_logits(logits, target)
            tl += loss.item()
            ta += acc
            steps += 1
            if tqdm is not None:
                pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc * 100:.2f}%")

        train_loss = tl / max(steps, 1)
        train_acc = ta / max(steps, 1)

        # 验证阶段
        val = evaluate(model, val_loader, criterion, device, True)
        val_loss, val_acc = val["loss"], val["acc"]

        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        scheduler.step(val_loss)
        print(f"Epoch {epoch}: Train {train_loss:.3f} Acc {train_acc * 100:.2f}% | "
              f"Val {val_loss:.3f} Acc {val_acc * 100:.2f}%")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

        # 保存历史并绘图
        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)

        plot_training_curves(hist, out_dir)
        # === Early stopping 检查 ===
        early_stopper.step(val_loss, epoch)
        if early_stopper.should_stop:
            print(f"[EarlyStopping] Stop training at epoch {epoch}. "
                  f"Best epoch was {early_stopper.best_epoch} with val_loss={early_stopper.best_loss:.4f}")
            break

    print(f"[Done] Training complete. Curves saved in {out_dir}")


def _strip_special(ids, ignore_ids):
    return [t for t in ids if t not in ignore_ids]


def _truncate_at_eos(ids, eos_id):
    if eos_id is None:
        return ids
    out = []
    for t in ids:
        if t == eos_id:
            break
        out.append(t)
    return out


def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _bleu_corpus(list_of_refs, list_of_hyps, max_n=4):
    # Papineni BLEU with smoothing: add-1 for n-gram precisions with zero counts
    hyp_len = 0
    ref_len = 0
    p_ns = []
    for n in range(1, max_n + 1):
        match = 0
        total = 0
        for refs, hyp in zip(list_of_refs, list_of_hyps):
            hyp_ngrams = _ngram_counts(hyp, n)
            max_ref_ngrams = Counter()
            # For corpus BLEU, choose a single reference length closest to hyp
            # and clip n-grams by the max across refs
            for r in refs:
                r_ngrams = _ngram_counts(r, n)
                for k, v in r_ngrams.items():
                    if max_ref_ngrams[k] < v:
                        max_ref_ngrams[k] = v
            overlap = {k: min(v, max_ref_ngrams.get(k, 0)) for k, v in hyp_ngrams.items()}
            match += sum(overlap.values())
            total += sum(hyp_ngrams.values())
        # add-1 smoothing
        p_ns.append((match + 1.0) / (total + 1.0))

    # brevity penalty (use closest reference lengths)
    for refs, hyp in zip(list_of_refs, list_of_hyps):
        hyp_len += len(hyp)
        if len(refs) == 1:
            ref_len += len(refs[0])
        else:
            # closest ref length
            ref_lens = [len(r) for r in refs]
            ref_len += min(ref_lens, key=lambda rl: (abs(rl - len(hyp)), rl))

    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - float(ref_len) / float(hyp_len))
    bleu = bp * math.exp(sum((1.0 / max_n) * math.log(p) for p in p_ns))
    return bleu


def _lcs(a, b):
    # O(len(a)*len(b)) DP
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                row[j] = prev[j - 1] + 1
            else:
                row[j] = max(prev[j], row[j - 1])
    return dp[-1][-1]


def _rouge_l_f1(list_of_refs, list_of_hyps):
    # average F1 over corpus
    f_sum = 0.0
    cnt = 0
    for refs, hyp in zip(list_of_refs, list_of_hyps):
        # If multiple refs, take the best ROUGE-L among them
        best_f = 0.0
        for r in refs:
            if len(hyp) == 0 and len(r) == 0:
                f = 1.0
            elif len(hyp) == 0 or len(r) == 0:
                f = 0.0
            else:
                l = _lcs(r, hyp)
                prec = l / len(hyp) if len(hyp) else 0.0
                rec = l / len(r) if len(r) else 0.0
                if prec + rec == 0:
                    f = 0.0
                else:
                    f = 2 * prec * rec / (prec + rec)
            if f > best_f:
                best_f = f
        f_sum += best_f
        cnt += 1
    return f_sum / cnt if cnt else 0.0


def _repetition_rate(tokens, n=2):
    """Return repetition rate for a single sequence at n-gram level."""
    if len(tokens) < n:
        return 0.0
    counts = _ngram_counts(tokens, n)
    total = sum(counts.values())
    unique = len(counts)
    if total == 0:
        return 0.0
    return 1.0 - (unique / total)


def _batch_repetition(seqs):
    # Report unigram and bigram repetition (mean over samples)
    if not seqs:
        return 0.0, 0.0
    r1 = sum(_repetition_rate(s, 1) for s in seqs) / len(seqs)
    r2 = sum(_repetition_rate(s, 2) for s in seqs) / len(seqs)
    return r1, r2


def _block_no_repeat_ngram(next_logits, histories, no_repeat_ngram_size, vocab_size):
    """
    next_logits: (B, V) logits of next step
    histories: list[list[int]] current decoded (without BOS), per batch
    In-place set -inf for tokens that would form a repeated n-gram.
    """
    if no_repeat_ngram_size is None or no_repeat_ngram_size < 2:
        return
    B = len(histories)
    n = no_repeat_ngram_size
    neg_inf = -1e9
    for b in range(B):
        hist = histories[b]
        if len(hist) < n - 1:
            continue
        # map (n-1)-gram prefix -> set of blocked next tokens
        banned = {}
        for i in range(len(hist) - n + 1):
            prefix = tuple(hist[i:i + n - 1])
            nxt = hist[i + n - 1]
            banned.setdefault(prefix, set()).add(nxt)
        prefix = tuple(hist[-(n - 1):])
        if prefix in banned:
            for t in banned[prefix]:
                next_logits[b, t] = neg_inf
        # also discourage PAD (rarely desirable mid-seq)
        # handled separately below if needed


def greedy_decode(model, src, src_padmask, bos_id, eos_id, pad_id,
                  max_len=64, min_len=1, no_repeat_ngram_size=3, device=None):
    """
    Batch greedy decoding (teacher-forcing-free).
    Returns: List[List[int]] (pred seqs, without BOS and cut at EOS)
    """
    device = device or src.device
    B = src.size(0)
    # start with BOS
    dec = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    histories = [[] for _ in range(B)]  # decoded tokens excluding BOS
    for step in range(1, max_len + 1):
        dec_padmask = dec.eq(pad_id)  # (B, L)
        logits = model(src, dec, src_key_padding_mask=src_padmask, tgt_key_padding_mask=dec_padmask)
        step_logits = logits[:, -1, :]  # (B, V)

        # forbid PAD
        step_logits[:, pad_id] = -1e9
        # forbid EOS if length < min_len
        if eos_id is not None and step < min_len:
            step_logits[:, eos_id] = -1e9
        # no-repeat-ngram
        _block_no_repeat_ngram(step_logits, histories, no_repeat_ngram_size, step_logits.size(-1))

        next_tok = step_logits.argmax(dim=-1)  # (B,)

        # once finished, keep emitting PAD to stabilize shape
        if eos_id is not None:
            next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)

        # update histories / finished
        for b in range(B):
            if not finished[b]:
                t = int(next_tok[b].item())
                if eos_id is not None and t == eos_id:
                    finished[b] = True
                else:
                    histories[b].append(t)

        # append
        dec = torch.cat([dec, next_tok.unsqueeze(1)], dim=1)

        # early stop: all finished 且 已达最小长度
        if finished.all() and step >= min_len:
            break

    # truncate at EOS (already handled by histories) and return
    return histories


def _decode_ids(ids, tok, pad_id, bos_id, eos_id):
    """
    将一串token id解码成字符串。
    优先使用 tok.decode；否则用字节级回退（去掉PAD/BOS/EOS后bytes->utf8）。
    """
    if hasattr(tok, "decode"):
        return tok.decode(ids)
    # fallback：去特殊符号，再按utf-8解码
    specials = set(x for x in (pad_id, bos_id, eos_id) if x is not None)
    clean = [i for i in ids if i not in specials]
    try:
        return bytes(clean).decode("utf-8", errors="ignore")
    except Exception:
        return "".join(chr(i) for i in clean if 32 <= i < 127)


@torch.no_grad()
def test(cfg, ckpt=None, verbose=True, log_interval=50, print_samples=3, freq_topk=10,
         gen_enable=True, gen_max_len=None, gen_min_len=1, no_repeat_ngram_size=5,
         save_limit=1000):
    """
    评测两路：TF（teacher-forcing） + GEN（greedy生成）
    现在会：
      1) 打印若干条可读文本样例（pred/ref）
      2) 保存样例到 <training.output_dir>/samples.jsonl（上限 save_limit 条）
      3) 保存评测汇总到 <training.output_dir>/metrics.json
    """
    import time
    import numpy as np
    from collections import Counter
    import json

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 取 test_loader + tokenizer
    _, _, test_loader, tok = build_dataloaders(cfg)

    data_cfg = cfg.get("data", {})
    PAD_ID = data_cfg.get("pad_id", None)
    BOS_ID = data_cfg.get("bos_id", None)
    EOS_ID = data_cfg.get("eos_id", None)
    # 若配置里没填，自动从 tokenizer 取（SPM 会有）
    if PAD_ID is None and hasattr(tok, "pad_id"):
        PAD_ID = tok.pad_id
    if BOS_ID is None and hasattr(tok, "bos_id"):
        BOS_ID = tok.bos_id
    if EOS_ID is None and hasattr(tok, "eos_id"):
        EOS_ID = tok.eos_id

    if BOS_ID is None or EOS_ID is None:
        raise ValueError("For generative evaluation, please set data.bos_id and data.eos_id in your config.")
    ignore_ids = {PAD_ID, BOS_ID}
    gen_max_len = gen_max_len or int(cfg["data"]["max_tgt_len"])

    # 输出目录与样例/指标路径
    out_dir = cfg["training"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    samples_path = os.path.join(out_dir, "samples.jsonl")
    samples_txt_path = os.path.join(out_dir, "samples.txt")  # 可选：简洁可读版
    metrics_path = os.path.join(out_dir, "metrics.json")

    # build model
    mcfg = cfg["model"]
    model = TransformerSeq2Seq(
        vocab_size=tok.vocab_size, d_model=mcfg["d_model"], n_heads=mcfg["n_heads"],
        d_ff=mcfg["d_ff"], n_layers=mcfg["n_layers"], dropout=mcfg["dropout"],
        use_bias=True, positional_encoding=mcfg["positional_encoding"],
        max_seq_len=max(cfg["data"]["max_src_len"], cfg["data"]["max_tgt_len"]),
        tie_weights=mcfg["tie_weights"]
    ).to(device)

    ckpt = ckpt or os.path.join(out_dir, "best_model.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # accumulators (TF)
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    tf_hyps, tf_refs = [], []
    tf_rep_seqs = []

    # accumulators (GEN)
    gen_hyps, gen_refs = [], []
    gen_rep_seqs = []

    # diagnostics / logging
    pred_token_counter = Counter()
    hyp_len_list, ref_len_list = [], []

    if verbose:
        print(f"[Test] device={device}; PAD={PAD_ID}, BOS={BOS_ID}, EOS={EOS_ID}")
        print(f"[Test] checkpoint: {ckpt}")
        if gen_enable:
            print(
                f"[GEN] greedy with max_len={gen_max_len}, min_len={gen_min_len}, no_repeat_ngram_size={no_repeat_ngram_size}")

    num_batches = len(test_loader)
    t_loop = time.time()
    sample_buffer_tf = []
    sample_buffer_gen = []

    # 打开文件句柄（覆盖式写入新的实验样例）
    jsonl_f = open(samples_path, "w", encoding="utf-8")
    txt_f = open(samples_txt_path, "w", encoding="utf-8")
    saved = 0

    model.eval()
    pbar = iter_with_pbar(test_loader, total=len(test_loader), desc="Test", leave=False)
    for b_idx, batch in enumerate(pbar, start=1):
        # ----- move to device -----
        src = batch["src"].to(device)
        dec_in = batch["dec_in"].to(device)
        target = batch["target"].to(device)
        src_pad = batch["src_padmask"].to(device)
        dec_pad = batch["dec_padmask"].to(device)

        # ----- TF forward -----
        logits = model(src, dec_in, src_key_padding_mask=src_pad, tgt_key_padding_mask=dec_pad)
        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
        acc = accuracy_from_logits(logits, target)

        total_loss += loss.item()
        total_acc += acc
        steps += 1

        # TF argmax sequences
        tf_pred_ids = logits.argmax(dim=-1).detach().cpu()
        tgt_ids = target.detach().cpu()
        src_ids = batch["src"].cpu()  # 用于解码src文本
        B = tgt_ids.size(0)

        # ----- Generative decoding -----
        gen_out = None
        if gen_enable:
            gen_out = greedy_decode(
                model, src=src, src_padmask=src_pad, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
                max_len=gen_max_len, min_len=gen_min_len, no_repeat_ngram_size=no_repeat_ngram_size, device=device
            )

        for i in range(B):
            # 取各序列的 id
            hyp_tf_ids = tf_pred_ids[i].tolist()
            ref_ids = tgt_ids[i].tolist()
            src_i_ids = src_ids[i].tolist()
            hyp_gen_ids = gen_out[i] if (gen_enable and gen_out is not None) else None

            # 规整（去特殊并按EOS截断）
            hyp_tf = _truncate_at_eos(_strip_special(hyp_tf_ids, ignore_ids), EOS_ID)
            ref = _truncate_at_eos(_strip_special(ref_ids, ignore_ids), EOS_ID)
            src_clean = _truncate_at_eos(_strip_special(src_i_ids, ignore_ids), EOS_ID)
            if hyp_gen_ids is not None:
                hyp_gen = _truncate_at_eos(_strip_special(hyp_gen_ids, ignore_ids), EOS_ID)
            else:
                hyp_gen = None

            # 统计
            tf_hyps.append(hyp_tf)
            tf_refs.append([ref])
            tf_rep_seqs.append(hyp_tf)
            pred_token_counter.update(hyp_tf)
            hyp_len_list.append(len(hyp_tf))
            ref_len_list.append(len(ref))
            if gen_enable and hyp_gen is not None:
                gen_hyps.append(hyp_gen)
                gen_refs.append([ref])
                gen_rep_seqs.append(hyp_gen)

            # # ===== 文本解码 =====
            # src_text = _decode_ids(src_clean, tok, PAD_ID, BOS_ID, EOS_ID)
            # ref_text = _decode_ids(ref, tok, PAD_ID, BOS_ID, EOS_ID)
            # tf_pred_text = _decode_ids(hyp_tf, tok, PAD_ID, BOS_ID, EOS_ID)
            # gen_pred_text = _decode_ids(hyp_gen, tok, PAD_ID, BOS_ID, EOS_ID) if (gen_enable and hyp_gen is not None) else None
            decoded_tf = tok.decode(hyp_tf)
            decoded_gen = tok.decode(hyp_gen)
            decoded_src = tok.decode(src)
            decoded_ref = tok.decode(ref)
            # ===== 打印少量样例（控制台）=====
            if len(sample_buffer_tf) < print_samples:
                sample_buffer_tf.append((decoded_tf, decoded_ref, decoded_src))
            if gen_enable and len(sample_buffer_gen) < print_samples and decoded_gen is not None:
                sample_buffer_gen.append((decoded_gen, decoded_ref, decoded_src))

            # ===== 保存样例（文件）=====
            if saved < save_limit:
                rec = {
                    "src_ids": src_clean,
                    "ref_ids": ref,
                    "tf_pred_ids": hyp_tf,
                    "gen_pred_ids": hyp_gen if hyp_gen is not None else None,
                    "src_text": decoded_src,
                    "ref_text": decoded_ref,
                    "tf_pred_text": decoded_tf,
                    "gen_pred_text": decoded_gen,
                }
                jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # 纯文本版（更易读）
                txt_f.write("==== SAMPLE ====\n")
                txt_f.write(f"SRC : {decoded_src}\n")
                txt_f.write(f"REF : {decoded_ref}\n")
                txt_f.write(f"TF  : {decoded_tf}\n")
                if decoded_gen is not None:
                    txt_f.write(f"GEN : {decoded_gen}\n")
                txt_f.write("\n")
                saved += 1
        if tqdm is not None:
            pbar.set_postfix(tf_loss=f"{total_loss / steps:.4f}", tf_acc=f"{(total_acc / steps) * 100:.2f}%")
        # ----- periodic log -----
        if verbose and (b_idx % log_interval == 0 or b_idx == num_batches):
            elapsed = time.time() - t_loop
            print(
                f"[Test][{b_idx}/{num_batches}] TF avg_loss={total_loss / steps:.4f} TF avg_acc={total_acc / steps * 100:.2f}%  "
                f"({elapsed:.1f}s since last log)")
            t_loop = time.time()

    # 关闭样例文件
    jsonl_f.close()
    txt_f.close()

    # ----- aggregate TF metrics -----
    tf_loss = total_loss / steps if steps else 0.0
    tf_acc = total_acc / steps if steps else 0.0
    tf_bleu = _bleu_corpus(tf_refs, tf_hyps) if tf_hyps else 0.0
    tf_rouge = _rouge_l_f1(tf_refs, tf_hyps) if tf_hyps else 0.0
    tf_r1, tf_r2 = _batch_repetition(tf_rep_seqs) if tf_rep_seqs else (0.0, 0.0)

    print(f"[TF][Summary] Loss={tf_loss:.4f} | Acc={tf_acc * 100:.2f}% | "
          f"BLEU={tf_bleu * 100:.2f} | ROUGE-L(F1)={tf_rouge * 100:.2f} | "
          f"Repetition@1={tf_r1 * 100:.2f}% Repetition@2={tf_r2 * 100:.2f}%")

    # ----- aggregate GEN metrics -----
    gen_bleu = gen_rouge = gen_r1 = gen_r2 = 0.0
    if gen_enable and gen_hyps:
        gen_bleu = _bleu_corpus(gen_refs, gen_hyps)
        gen_rouge = _rouge_l_f1(gen_refs, gen_hyps)
        gen_r1, gen_r2 = _batch_repetition(gen_rep_seqs)
        print(f"[GEN][Summary] BLEU={gen_bleu * 100:.2f} | ROUGE-L(F1)={gen_rouge * 100:.2f} | "
              f"Repetition@1={gen_r1 * 100:.2f}% Repetition@2={gen_r2 * 100:.2f}%")

    # ----- diagnostics -----
    if verbose:
        dur = time.time() - t_start
        if hyp_len_list and ref_len_list:
            hyp_arr = np.array(hyp_len_list);
            ref_arr = np.array(ref_len_list)
            print("[Test][TF Lengths] "
                  f"pred_mean={hyp_arr.mean():.1f} pred_med={np.median(hyp_arr):.0f} "
                  f"| ref_mean={ref_arr.mean():.1f} ref_med={np.median(ref_arr):.0f}")

        if pred_token_counter:
            topk = pred_token_counter.most_common(freq_topk)
            total_pred_tok = sum(pred_token_counter.values())
            print(f"[Test][TF Top-{freq_topk} Pred Tokens]")
            for tid, cnt in topk:
                print(f"  id={tid:<4} count={cnt:<6} ratio={cnt / total_pred_tok:>.2%}")

        if sample_buffer_tf:
            print(f"[TF][Samples x{len(sample_buffer_tf)}] (text)")
            for idx, (pred_t, ref_t, src_t) in enumerate(sample_buffer_tf, 1):
                print(f"  --- TF Sample #{idx} ---")
                print(f"  SRC : {src_t[:200]}{' ...' if len(src_t) > 200 else ''}")
                print(f"  REF : {ref_t[:200]}{' ...' if len(ref_t) > 200 else ''}")
                print(f"  PRED: {pred_t[:200]}{' ...' if len(pred_t) > 200 else ''}")

        if gen_enable and sample_buffer_gen:
            print(f"[GEN][Samples x{len(sample_buffer_gen)}] (text)")
            for idx, (pred_t, ref_t, src_t) in enumerate(sample_buffer_gen, 1):
                print(f"  --- GEN Sample #{idx} ---")
                print(f"  SRC : {src_t[:200]}{' ...' if len(src_t) > 200 else ''}")
                print(f"  REF : {ref_t[:200]}{' ...' if len(ref_t) > 200 else ''}")
                print(f"  PRED: {pred_t[:200]}{' ...' if len(pred_t) > 200 else ''}")

        print(f"[Test] Done in {dur:.2f}s.")
        print(f"[Test] Samples saved: {samples_path}")
        print(f"[Test] Readable samples: {samples_txt_path}")

    # ----- 保存指标到文件 -----
    metrics = {
        "tf_loss": tf_loss,
        "tf_acc": tf_acc,
        "tf_bleu": tf_bleu,
        "tf_rouge_l_f1": tf_rouge,
        "tf_repetition@1": tf_r1,
        "tf_repetition@2": tf_r2,
    }
    if gen_enable:
        metrics.update({
            "gen_bleu": gen_bleu,
            "gen_rouge_l_f1": gen_rouge,
            "gen_repetition@1": gen_r1,
            "gen_repetition@2": gen_r2,
        })
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--ckpt", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.mode == "train":
        train(cfg)
    else:
        test(cfg, args.ckpt)


if __name__ == "__main__":
    main()
