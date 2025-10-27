# Minimal Transformer (from-scratch) Experiment

This repo contains a clean, educational PyTorch implementation of an **encoder-only Transformer** for character-level next-token prediction.
It implements **Multi-Head Self-Attention**, **Position-wise FFN**, **Residual + LayerNorm**, and **Sinusoidal Positional Encoding** — matching the requirements of your assignment.

## Features
- From-scratch modules: Scaled Dot-Product Attention, Multi-Head Self-Attention, FFN, Positional Encoding, Residual + LayerNorm.
- Config-driven training via YAML (no external datasets required).
- Tiny demo dataset included at `data/input.txt`.
- Reproducible runs, model checkpoints, and training curves.

## Quickstart

```bash
# (Optional) create env
conda create -n transformer python=3.10 -y
conda activate transformer

# Install deps
pip install -r requirements.txt

# Train (uses configs/base.yaml)
python train.py --config configs/base.yaml
```

After training, artifacts are saved to `runs/<run_name>/`:
- `model.pt`: trained weights
- `config.yaml`: resolved config
- `train_curve.png`: loss plot

## Evaluate / Generate
```bash
# Evaluate on validation split; also sample generated text
python eval.py --ckpt runs/exp1/model.pt --prompt "The meaning of life " --steps 200
```

## Files
- `transformer/modules.py` – attention, MHA, FFN, positional encoding, residual+LN
- `transformer/model.py` – `TransformerEncoderLM` (embeddings, N encoder layers, LM head)
- `transformer/data.py` – byte-level char dataset + splits, batching, masks
- `train.py` – training loop with AdamW, cosine schedule, gradient clipping
- `eval.py` – evaluation & text generation
- `configs/base.yaml` – hyperparameters
- `data/input.txt` – tiny sample corpus (you can replace with your own text)

## Notes
- This is an *educational*, minimal build. For larger corpora, consider enabling mixed precision (`--amp`) and increasing model size.
- The implementation intentionally avoids calling `nn.TransformerEncoder` to align with the "hand-rolled" requirement.
