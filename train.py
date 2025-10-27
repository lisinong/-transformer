import argparse
import os
import random
import yaml

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from transformer.data import get_loaders
from transformer.model import TransformerEncoderLM


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str):
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
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
    run_name = cfg["run_name"]
    ckpt_dir = os.path.join(cfg["train"]["ckpt_dir"], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    tr_loader, va_loader = get_loaders(
        cfg["data"]["path"],
        block_size=int(cfg["data"]["block_size"]),
        batch_size=int(cfg["data"]["batch_size"]),
        train_frac=float(cfg["data"]["train_frac"]),
        num_workers=int(cfg["data"]["num_workers"]),
        seed=int(cfg["seed"]),
    )

    # Model
    model = TransformerEncoderLM(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
        use_bias=cfg["model"]["use_bias"],
    ).to(device)

    # Optimizer & scheduler
    opt = AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]), betas=tuple(cfg["optim"]["betas"]),
                weight_decay=cfg["optim"]["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=cfg["train"]["max_steps"] - cfg["sched"]["warmup_steps"],
                              eta_min=cfg["sched"]["min_lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")

    def lr_warmup(step):
        ws = cfg["sched"]["warmup_steps"]
        base = cfg["optim"]["lr"]
        if step < ws:
            return base * (step + 1) / max(1, ws)
        return None  # use scheduler thereafter

    best_val = float("inf")
    history = []
    global_step = 0

    pbar = tqdm(total=cfg["train"]["max_steps"], desc="training", ncols=100)
    while global_step < cfg["train"]["max_steps"]:
        for xb, yb in tr_loader:
            model.train()
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(xb)  # [B,T,V]
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                opt.step()

            # LR scheduling
            if lr_warmup(global_step) is not None:
                for g in opt.param_groups:
                    g["lr"] = lr_warmup(global_step)
            else:
                sched.step()

            global_step += 1
            pbar.update(1)

            if global_step % cfg["train"]["log_every"] == 0:
                # quick val
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for xvb, yvb in va_loader:
                        xvb = xvb.to(device);
                        yvb = yvb.to(device)
                        logits = model(xvb)
                        vloss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yvb.view(-1))
                        val_losses.append(vloss.item())
                val_loss = sum(val_losses) / max(1, len(val_losses))
                history.append({"step": global_step, "train_loss": loss.item(), "val_loss": val_loss})
                tqdm.write(
                    f"step {global_step} | train {loss.item():.4f} | val {val_loss:.4f} | lr {opt.param_groups[0]['lr']:.2e}")

            if global_step % cfg["train"]["eval_every"] == 0:
                # save checkpoint
                torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(ckpt_dir, "model.pt"))
                yaml.safe_dump(cfg, open(os.path.join(ckpt_dir, "config.yaml"), "w"))
                # plot curve
                try:
                    save_curve(history, os.path.join(ckpt_dir, "train_curve.png"))
                except Exception:
                    pass

            if global_step >= cfg["train"]["max_steps"]:
                break

    # final save
    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(ckpt_dir, "model.pt"))
    yaml.safe_dump(cfg, open(os.path.join(ckpt_dir, "config.yaml"), "w"))
    try:
        save_curve(history, os.path.join(ckpt_dir, "train_curve.png"))
    except Exception:
        pass
    pbar.close()
    print(f"Done. Checkpoints in {ckpt_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    train(cfg)


if __name__ == "__main__":
    main()
