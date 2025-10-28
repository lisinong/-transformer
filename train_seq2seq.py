import argparse
import os
import random
import yaml

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformer.data import get_loaders_from_ted, PAD
from transformer.model_seq2seq import TransformerSeq2Seq


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


def save_curve(history, out_png):
    import matplotlib.pyplot as plt
    steps = [h["step"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    plt.figure()
    plt.plot(steps, train_loss, label="train")
    plt.plot(steps, val_loss, label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)


def train(cfg):
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    run = cfg["run_name"]
    ckpt_dir = os.path.join(cfg["train"]["ckpt_dir"], run)
    os.makedirs(ckpt_dir, exist_ok=True)

    tr_loader, va_loader, te_loader, bos, eos, pad = get_loaders_from_ted(
        zip_path=cfg["data"].get("zip_path", None),
        transcripts_csv=cfg["data"].get("transcripts_csv", None),
        meta_csv=cfg["data"].get("meta_csv", None),
        src_field=cfg["data"].get("src_field", "transcript"),
        tgt_field=cfg["data"].get("tgt_field", "title"),
        max_src_len=cfg["data"]["max_src_len"],
        max_tgt_len=cfg["data"]["max_tgt_len"],
        min_src_chars=cfg["data"].get("min_src_chars", 64),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
        train_frac=cfg["data"].get("train_frac", 0.96),
        valid_frac=cfg["data"].get("valid_frac", 0.02),
        test_frac=cfg["data"].get("test_frac", 0.02),
    )

    model_cfg = dict(cfg["model"])
    # byte-level vocab: 0..255 plus BOS/EOS/PAD mapped to 1/2/0, so keep vocab_size >= 256 or set as config
    model = TransformerSeq2Seq(**model_cfg).to(device)

    opt = AdamW(model.parameters(), lr=cfg["optim"]["lr"], betas=tuple(cfg["optim"]["betas"]),
                weight_decay=cfg["optim"]["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=cfg["train"]["max_steps"] - cfg["sched"]["warmup_steps"],
                              eta_min=cfg["sched"]["min_lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")

    hist = [];
    step = 0
    pbar = tqdm(total=cfg["train"]["max_steps"], desc="training", ncols=100)
    while step < cfg["train"]["max_steps"]:
        for src, tin, tout, src_kpm, tin_kpm in tr_loader:
            model.train()
            src, tin, tout = src.to(device), tin.to(device), tout.to(device)
            src_kpm, tin_kpm = src_kpm.to(device), tin_kpm.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(src, tin, src_key_padding_mask=src_kpm, tgt_key_padding_mask=tin_kpm)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tout.view(-1), ignore_index=PAD)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(opt);
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                opt.step()

            if step < cfg["sched"]["warmup_steps"]:
                lr = cfg["optim"]["lr"] * (step + 1) / max(1, cfg["sched"]["warmup_steps"])
                for g in opt.param_groups:
                    g["lr"] = lr
            else:
                sched.step()

            step += 1
            pbar.update(1)

            if step % cfg["train"]["log_every"] == 0:
                model.eval()
                vals = []
                with torch.no_grad():
                    for s2, ti2, to2, sk2, tk2 in va_loader:
                        s2, ti2, to2 = s2.to(device), ti2.to(device), to2.to(device)
                        sk2, tk2 = sk2.to(device), tk2.to(device)
                        logits2 = model(s2, ti2, src_key_padding_mask=sk2, tgt_key_padding_mask=tk2)
                        vloss = nn.functional.cross_entropy(logits2.view(-1, logits2.size(-1)), to2.view(-1),
                                                            ignore_index=PAD)
                        vals.append(vloss.item())
                v = sum(vals) / max(1, len(vals))
                hist.append({"step": step, "train_loss": float(loss.item()), "val_loss": float(v)})
                pbar.write(f"step {step} | train {loss.item():.4f} | val {v:.4f}")

            if step % cfg["train"]["eval_every"] == 0:
                torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(ckpt_dir, "model.pt"))
                import yaml as _y
                _y.safe_dump(cfg, open(os.path.join(ckpt_dir, "config.yaml"), "w"))
                try:
                    save_curve(hist, os.path.join(ckpt_dir, "train_curve.png"))
                except Exception:
                    pass

            if step >= cfg["train"]["max_steps"]: break

    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(ckpt_dir, "model.pt"))
    import yaml as _y
    _y.safe_dump(cfg, open(os.path.join(ckpt_dir, "config.yaml"), "w"))
    try:
        save_curve(hist, os.path.join(ckpt_dir, "train_curve.png"))
    except Exception:
        pass
    pbar.close()
    print(f"Done. Checkpoints in {ckpt_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base_seq2seq.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    train(cfg)


if __name__ == "__main__":
    main()
