
import argparse
import torch

from transformer.model import TransformerEncoderLM


def load_ckpt(ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    cfg = ckpt.get("config", None)
    model = TransformerEncoderLM(**cfg["model"]).to(map_location if isinstance(map_location, torch.device) else torch.device("cpu"))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg

@torch.no_grad()
def generate(model, prompt: bytes, steps: int = 200):
    device = next(model.parameters()).device
    idx = torch.tensor(list(prompt), dtype=torch.long, device=device)[None, :]  # [1,T]
    for _ in range(steps):
        logits = model(idx)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1,1]
        idx = torch.cat([idx, next_token], dim=1)
    return bytes(idx[0].tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to runs/<run>/model.pt")
    ap.add_argument("--prompt", type=str, default="Hello, ")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_ckpt(args.ckpt, map_location=device)
    out = generate(model, args.prompt.encode("utf-8"), steps=args.steps)
    print(out.decode("utf-8", errors="ignore"))

if __name__ == "__main__":
    main()
