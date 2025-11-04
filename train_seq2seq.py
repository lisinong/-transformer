import argparse
import os
import random
import sys
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from transformer.data import _load_ted_frames, get_loaders_from_ted, _merge_ted, PAD, BOS, EOS, clean_text
from transformer.model_seq2seq import TransformerSeq2Seq
from torch.optim import AdamW, Adam, SGD, RMSprop, Adagrad
import sentencepiece as spm


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


def build_optimizer(model, cfg):
    o = cfg["optim"]
    name = o.get("name", "adamw").lower()
    lr = float(o.get("lr", 5e-4))
    wd = float(o.get("weight_decay", 0.01))
    eps = float(o.get("eps", 1e-8))

    if name == "adamw":
        betas = tuple(o.get("betas", [0.9, 0.98]))
        return AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd, eps=eps)
    elif name == "adam":
        betas = tuple(o.get("betas", [0.9, 0.999]))
        return Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd, eps=eps)
    elif name == "sgd":
        momentum = float(o.get("momentum", 0.9))
        nesterov = bool(o.get("nesterov", True))
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
    elif name == "rmsprop":
        momentum = float(o.get("momentum", 0.0))
        alpha = float(o.get("alpha", 0.99))
        return RMSprop(model.parameters(), lr=lr, momentum=momentum, alpha=alpha, eps=eps, weight_decay=wd,
                       centered=False)
    elif name == "adagrad":
        lr_decay = float(o.get("lr_decay", 0.0))
        return Adagrad(model.parameters(), lr=lr, weight_decay=wd, lr_decay=lr_decay, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str):
    if pref == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def save_curve(history, out_png, is_epoch=True):  # <--- 【改动点 1】: 增加 is_epoch 参数，让绘图函数更灵活
    import matplotlib.pyplot as plt
    x_axis = "epoch" if is_epoch else "step"
    x_values = [h[x_axis] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    plt.figure()
    plt.plot(x_values, train_loss, label="train")
    plt.plot(x_values, val_loss, label="val")
    plt.xlabel(x_axis)
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)


def prepare_tokenizer(cfg):
    """
    若 spm_model 不存在，则使用 transcript + description + title 共同训练一个新的 SentencePiece 模型。
    训练语料不做过度清洗，仅做基础空白规范化，避免破坏空格/标点的统计。
    """
    import os, html, re
    import pandas as pd
    import sentencepiece as spm

    spm_model_path = cfg["data"]["spm_model_path"]
    spm_vocab_path = spm_model_path.replace(".model", ".vocab")
    if os.path.exists(spm_model_path) and os.path.exists(spm_vocab_path):
        print(f"Tokenizer model found at '{spm_model_path}'. Skipping training.")
        return

    transcripts_csv = cfg["data"].get("transcripts_csv")
    meta_csv = cfg["data"].get("meta_csv")
    assert transcripts_csv and meta_csv, "Need both transcripts_csv and meta_csv to build SPM."

    print(f"Tokenizer model not found. Starting training...")
    print("  -> Creating corpus file for training...")

    df_t = pd.read_csv(transcripts_csv)
    df_m = pd.read_csv(meta_csv)

    def normalize_text(s: str) -> str:
        if not isinstance(s, str):
            s = "" if s is None else str(s)
        s = html.unescape(s)
        # 只做空白规范化，避免丢空格或标点
        s = re.sub(r"\s+", " ", s).strip()
        return s

    corpus_path = "temp_corpus_for_spm.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        # 1) transcripts（长文本是关键）
        if "transcript" in df_t.columns:
            for text in df_t["transcript"].dropna():
                f.write(normalize_text(text) + "\n")
        # 2) descriptions
        if "description" in df_m.columns:
            for text in df_m["description"].dropna():
                f.write(normalize_text(text) + "\n")
        # 3) titles
        if "title" in df_m.columns:
            for text in df_m["title"].dropna():
                f.write(normalize_text(text) + "\n")

    # 训练参数
    vocab_size = cfg["model"].get("vocab_size", 8000)
    bos_id = cfg["data"].get("bos", 256)
    eos_id = cfg["data"].get("eos", 257)
    pad_id = cfg["data"].get("pad", 258)
    model_prefix = spm_model_path.replace(".model", "")

    # 建议使用 unigram + 高覆盖率；UNK 用默认 0（不要设成 3）
    spm_command = (
        f'--input={corpus_path} '
        f'--model_prefix={model_prefix} '
        f'--model_type=unigram '
        f'--vocab_size={vocab_size} '
        f'--character_coverage=0.9995 '
        f'--shuffle_input_sentence=true '
        f'--input_sentence_size=1000000 '
        f'--bos_id={bos_id} --eos_id={eos_id} --pad_id={pad_id} --unk_id=0'
    )

    print("  -> Training SentencePiece model...")
    spm.SentencePieceTrainer.train(spm_command)
    os.remove(corpus_path)
    print(f"Tokenizer model trained and saved as '{spm_model_path}' and '{spm_vocab_path}'.")
    print("-" * 50)



def train(cfg):
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    prepare_tokenizer(cfg)
    run = cfg["run_name"]
    ckpt_dir = os.path.join(cfg["train"]["ckpt_dir"], run)
    os.makedirs(ckpt_dir, exist_ok=True)

    tr_loader, va_loader, te_loader, bos, eos, pad, sp_processor = get_loaders_from_ted(
        spm_model_path=cfg["data"]["spm_model_path"],
        zip_path=cfg["data"].get("zip_path", None),
        transcripts_csv=cfg["data"].get("transcripts_csv", None),
        meta_csv=cfg["data"].get("meta_csv", None),
        src_field=cfg["data"].get("src_field", "transcript"),
        tgt_field=cfg["data"].get("tgt_field", "description"),
        max_src_len=cfg["data"]["max_src_len"],
        max_tgt_len=cfg["data"]["max_tgt_len"],
        min_src_chars=cfg["data"].get("min_src_chars", 64),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
        train_frac=cfg["data"].get("train_frac", 0.96),
        valid_frac=cfg["data"].get("valid_frac", 0.02),
        test_frac=cfg["data"].get("test_frac", 0.02),
        bos=cfg["data"].get("bos", BOS),
        eos=cfg["data"].get("eos", EOS),
        pad=cfg["data"].get("pad", PAD)
    )
    vocab_size = sp_processor.vocab_size()
    print(f"Using actual vocab size from tokenizer: {vocab_size}")
    cfg["model"]["vocab_size"] = vocab_size

    model_cfg = dict(cfg["model"])
    label_smoothing = model_cfg.pop('label_smoothing', 0.0)
    model = TransformerSeq2Seq(**model_cfg).to(device)
    opt = build_optimizer(model, cfg)

    # --- 【改动点 2】: 学习率调度器配置修改 ---
    # 从配置文件读取 num_epochs，并计算总步数
    num_epochs = cfg["train"].get("num_epochs", 100)  # 默认为100个epoch
    steps_per_epoch = len(tr_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = cfg["sched"].get("warmup_steps", 4000)
    sched_kind = cfg["sched"].get("scheduler", "cosine").lower()

    # 确保 Cosine LR 的 T_max 基于总步数计算
    if sched_kind == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        # T_max 应该是 warmup 之后剩余的总步数
        sched = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=cfg["sched"]["min_lr"])
        use_noam = False
    elif sched_kind == "noam":
        sched = NoamLR(opt, d_model=cfg["model"]["d_model"], warmup_steps=warmup_steps)
        use_noam = True
    else:
        raise ValueError(f"Unknown scheduler: {sched_kind}")

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")
    accum_steps = int(cfg["train"].get("accum_steps", 1))

    ce_loss = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing)
    hist = []
    step = 0
    best_val_loss = float('inf')  # <--- 【改动点 3】: 用于 Early Stopping

    # --- 【改动点 4】: 主训练循环改为按 epoch ---
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)

        epoch_train_losses = []  # 用于记录当前epoch的训练损失

        for src, tin, tout, src_kpm, tin_kpm in pbar:
            src, tin, tout = src.to(device), tin.to(device), tout.to(device)
            src_kpm, tin_kpm = src_kpm.to(device), tin_kpm.to(device)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(src, tin, src_key_padding_mask=src_kpm, tgt_key_padding_mask=tin_kpm)
                loss = ce_loss(logits.view(-1, logits.size(-1)), tout.view(-1))
                loss = loss / accum_steps

            epoch_train_losses.append(loss.item() * accum_steps)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积和优化器步骤
            if ((step + 1) % accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

                # 学习率更新 (仍然按step更新)
                current_step_for_lr = step // accum_steps
                if use_noam:
                    sched.step()
                else:
                    if current_step_for_lr < warmup_steps:
                        lr = cfg["optim"]["lr"] * (current_step_for_lr + 1) / max(1, warmup_steps)
                        for g in opt.param_groups: g["lr"] = lr
                    else:
                        sched.step()

            step += 1
            pbar.set_postfix(lr=opt.param_groups[0]['lr'], loss=epoch_train_losses[-1])

        # --- 【改动点 5】: 每个 epoch 结束后进行验证和保存 ---
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)

        # 验证逻辑
        model.eval()
        val_losses = []
        with torch.no_grad():
            for s2, ti2, to2, sk2, tk2 in va_loader:
                s2, ti2, to2 = s2.to(device), ti2.to(device), to2.to(device)
                sk2, tk2 = sk2.to(device), tk2.to(device)
                logits2 = model(s2, ti2, src_key_padding_mask=sk2, tgt_key_padding_mask=tk2)
                vloss = ce_loss(logits2.view(-1, logits2.size(-1)), to2.view(-1)).item()
                val_losses.append(vloss)
        avg_val_loss = sum(val_losses) / max(1, len(val_losses))

        hist.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        print(f"Epoch {epoch + 1} summary | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        # 保存最佳模型 (Early Stopping)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> New best model saved with val_loss: {best_val_loss:.4f}")

        # 定期保存最新模型和训练曲线
        if (epoch + 1) % cfg["train"].get("save_every_epochs", 5) == 0:  # 默认为每5个epoch保存一次
            torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt"))
            try:
                save_curve(hist, os.path.join(ckpt_dir, "train_curve.png"), is_epoch=True)
            except Exception:
                pass

    # 训练结束后，保存最终模型和配置
    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(ckpt_dir, "final.pt"))
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    try:
        save_curve(hist, os.path.join(ckpt_dir, "train_curve.png"), is_epoch=True)
    except Exception:
        pass

    print(f"Done. Best validation loss: {best_val_loss:.4f}. Checkpoints in {ckpt_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base_seq2seq.yaml")
    args = ap.parse_args()
    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
