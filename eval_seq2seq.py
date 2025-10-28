\
import argparse, torch, yaml
from transformer.model_seq2seq import TransformerSeq2Seq, PAD, EOS, BOS
from transformer.data_seq2seq import encode_line, pack_with_specials

@torch.no_grad()
def greedy_decode(model, src_ids, max_len=128, device="cpu"):
    src = pack_with_specials(src_ids, max_len, add_bos=True, add_eos=True)
    src = torch.tensor(src, dtype=torch.long, device=device)[None, :]
    src_kpm = (src == PAD)
    tgt = torch.tensor([[BOS]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        logits = model(src, tgt, src_key_padding_mask=src_kpm, tgt_key_padding_mask=(tgt == PAD))
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
        if int(next_token.item()) == EOS: break
    return tgt[0].tolist()

def detok(ids):
    out = []
    for i in ids:
        if i in (BOS, EOS, PAD): continue
        out.append(i)
    return bytes(out).decode("utf-8", errors="ignore")

def load_ckpt(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt["config"]
    model = TransformerSeq2Seq(**cfg["model"]).to(map_location if isinstance(map_location, torch.device) else torch.device("cpu"))
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_ckpt(args.ckpt, map_location=device)
    ids = encode_line(args.src)
    out_ids = greedy_decode(model, ids, max_len=args.max_len, device=device)
    print(detok(out_ids))

if __name__ == "__main__":
    main()
