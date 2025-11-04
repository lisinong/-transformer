import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
import sentencepiece as spm
import math
# 确保从你修改后的 data.py 和 model_seq2seq.py 导入
from transformer.data import get_loaders_from_ted, PAD
from transformer.model_seq2seq import TransformerSeq2Seq


def resolve_device(pref: str):
    """自动选择设备"""
    if pref == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)

# === NEW: safe detokenize utilities ===
# === safe detokenize ===
def ids_to_text(sp, ids, bos, eos, pad):
    arr = [int(x) for x in ids]
    # 去 PAD
    arr = [x for x in arr if x != pad]
    # 去头部 BOS
    if arr and arr[0] == bos:
        arr = arr[1:]
    # 截断到 EOS
    if eos in arr:
        arr = arr[:arr.index(eos)]
    # 只用 SentencePiece 原生解码；不要做任何 replace / lower / 正则
    return sp.decode_ids(arr)

def batch_ids_to_text(sp, ids_batch, bos, eos, pad):
    return [ids_to_text(sp, row, bos, eos, pad) for row in ids_batch]


def beam_search(model, src, src_kpm, bos, eos, pad, max_new_tokens, beam_size, device="cpu", lp_alpha=0.6):
    """
    为单个样本执行 Beam Search。仅支持 batch_size=1。
    """
    assert src.size(0) == 1, "Beam search implementation only supports batch_size=1"
    min_len = 5
    no_repeat_ngram_size = 3
    repetition_penalty = 1.2
    log_pen = math.log(repetition_penalty)

    beams = [(torch.tensor([[bos]], dtype=torch.long, device=device), 0.0)]  # (sequence, logprob)
    with torch.no_grad():
        encoder_out = model.encode(src, src_key_padding_mask=src_kpm)

        for step in range(max_new_tokens):
            new_beams = []

            # 先把已经到 EOS 的 beam 原样带入（不再扩展）
            ended = [(seq, score) for (seq, score) in beams if seq[0, -1].item() == eos]
            alive  = [(seq, score) for (seq, score) in beams if seq[0, -1].item() != eos]
            new_beams.extend(ended)

            # 若全结束就提前退出
            if len(alive) == 0:
                beams = new_beams[:beam_size]
                break

            for seq, score in alive:
                # 解码一步（确保因果掩码）
                logits = model.decode(encoder_out, seq, None, None, True)
                logp = F.log_softmax(logits[:, -1, :], dim=-1)

                # 基础屏蔽
                logp[:, pad] = -float("inf")
                if seq.size(1) > 1:  # 除首 token 外禁止 BOS
                    logp[:, bos] = -float("inf")
                if step < min_len:    # 最小生成长度前禁止 EOS
                    logp[:, eos] = -float("inf")

                # 相邻重复禁：禁止上一个 token 再出现
                last_id = seq[0, -1].item()
                logp[0, last_id] = -float("inf")

                # repetition penalty：对已出现过的 token 统一降权
                prev_tokens = set(seq[0].tolist())
                for t in prev_tokens:
                    if t not in (bos, eos, pad):
                        logp[0, t] = logp[0, t] - log_pen

                # no-repeat n-gram
                if no_repeat_ngram_size > 0 and seq.size(1) >= no_repeat_ngram_size - 1:
                    n = no_repeat_ngram_size
                    hist = seq[0].tolist()
                    prefix2next = {}
                    for j in range(len(hist) - n + 1):
                        pre = tuple(hist[j:j+n-1])
                        nxt = hist[j+n-1]
                        prefix2next.setdefault(pre, set()).add(nxt)
                    cur_pre = tuple(hist[-(n-1):])
                    for banned in prefix2next.get(cur_pre, []):
                        logp[0, banned] = -float("inf")

                # 取 top-k
                topk = torch.topk(logp, beam_size, dim=-1)
                for k in range(beam_size):
                    next_id = topk.indices[0, k].view(1, 1)
                    new_seq = torch.cat([seq, next_id], dim=1)
                    new_score = score + float(topk.values[0, k])
                    new_beams.append((new_seq, new_score))

            # 剪枝
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

            # 全部到 EOS 也可退出
            if all(b[0][0, -1].item() == eos for b in beams):
                break

    # 长度惩罚选最优
    best = max(beams, key=lambda x: x[1] / ((5 + x[0].size(1)) / 6) ** lp_alpha)
    return best[0]


def eval_split(model, cfg, split, decode, max_new_tokens=64, beam_size=4, device="cpu"):
    """
    在指定的数据集 split 上评估模型。
    """
    # 1. 加载与训练时完全一致的分词器
    sp = spm.SentencePieceProcessor()
    spm_model_path = cfg["data"]["spm_model_path"]
    if not os.path.exists(spm_model_path):
        raise FileNotFoundError(
            f"SentencePiece model not found at '{spm_model_path}'. Please run train.py first to generate it.")
    sp.load(spm_model_path)
    print(f"Loaded SentencePiece model from {spm_model_path} with vocab size {sp.vocab_size()}")

    bos, eos, pad = sp.bos_id(), sp.eos_id(), sp.pad_id()
    print(f"[debug] Special IDs -> BOS={bos}, EOS={eos}, PAD={pad}")

    # 2. 使用与训练时相同的函数加载数据，确保数据处理一致
    tr_loader, va_loader, te_loader, _, _, _, _ = get_loaders_from_ted(
        spm_model_path=spm_model_path,
        zip_path=cfg["data"].get("zip_path", None),
        transcripts_csv=cfg["data"].get("transcripts_csv", None),
        meta_csv=cfg["data"].get("meta_csv", None),
        src_field=cfg["data"].get("src_field", "transcript"),
        tgt_field=cfg["data"].get("tgt_field", "description"),
        max_src_len=cfg["data"]["max_src_len"],
        max_tgt_len=cfg["data"]["max_tgt_len"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        seed=cfg["seed"]
    )

    loader_map = {"train": tr_loader, "valid": va_loader, "test": te_loader}
    loader = loader_map.get(split)
    if loader is None:
        raise ValueError(f"Invalid split '{split}'. Choose from {list(loader_map.keys())}")

    print(
        f"[debug] split={split}, batch_size={cfg['data']['batch_size']}, max_src_len={cfg['data']['max_src_len']}, max_tgt_len={cfg['data']['max_tgt_len']}")

    model.eval()
    all_hyps, all_refs = [], []

    pbar = tqdm(loader, desc=f"Evaluating {split} set", ncols=120)
    for src, tin, tout, src_kpm, tin_kpm in pbar:
        src, src_kpm = src.to(device), src_kpm.to(device)

        # --- 模型生成 ---
        if decode == 'greedy':
            ys = model.generate(src, src_key_padding_mask=src_kpm, max_new_tokens=max_new_tokens, bos=bos, eos=eos,
                                pad=pad)
        elif decode == 'beam':
            ys_list = []
            for i in range(src.size(0)):
                # 对 batch 中的每个样本单独调用 beam_search
                y1 = beam_search(model, src[i:i + 1], src_kpm[i:i + 1], bos, eos, pad,
                                 max_new_tokens, beam_size, device)
                ys_list.append(y1.squeeze(0))

            # 将不同长度的序列填充为同一长度的批次
            ys = torch.nn.utils.rnn.pad_sequence(ys_list, batch_first=True, padding_value=pad)
        else:
            raise ValueError(f"Unknown decode method: {decode}")

        # --- 3. 使用 SentencePiece 正确解码为文本 ---
        # hyps_text = sp.decode(ys.tolist())
        # refs_text = sp.decode(tout.tolist())
        hyps_text = batch_ids_to_text(sp, ys.tolist(), bos, eos, pad)
        refs_text = batch_ids_to_text(sp, tout.tolist(), bos, eos, pad)
        all_hyps.extend(hyps_text)
        all_refs.extend(refs_text)
        # # 仅调试用：检查是否仍然存在可疑的 ".s" 或 ",s" 模式
        # if any((".s" in x or ",s" in x) for x in hyps_text[:10]):
        #     raise RuntimeError("Detected suspicious '.s' / ',s' in hypothesis text. "
        #                "Make sure you are using ids_to_text(sp, ids, bos, eos, pad) "
        #                "and NOT doing any piece-level replace or lower().")

        # --- 调试输出 ---
        # 仅在非 tqdm 环境或需要详细日志时打开
        # print(f"\n[debug] hyp0_ids[:20]={ys[0][:20].tolist()}")
        # print(f"[debug] hyp0_txt[:120]='{hyps_text[0][:120]}'")
        # print(f"[debug] ref0_txt[:120]='{refs_text[0][:120]}'")

    # --- ROUGE 分数计算 ---
    print("\nCalculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for hyp, ref in zip(all_hyps, all_refs):
        # 确保不会因为空字符串导致 rouge 计算出错
        hyp = hyp if hyp.strip() else "empty"
        ref = ref if ref.strip() else "empty"
        aggregator.add_scores(scorer.score(ref, hyp))

    result = aggregator.aggregate()
    print(f"\n[{split}] Final ROUGE scores for {len(all_hyps)} samples:")
    for key, value in result.items():
        print(
            f"  {key:<8}: F1={value.mid.fmeasure:.4f} (Precision={value.mid.precision:.4f}, Recall={value.mid.recall:.4f})")

    # 打印一些样本进行直观感受
    print("\n--- Sample Generations ---")
    for i in range(min(5, len(all_hyps))):
        print(f"Sample {i + 1}:")
        print(f"  REF: {all_refs[i]}")
        print(f"  HYP: {all_hyps[i]}")
        print("-" * 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to the .pt checkpoint file.")
    ap.add_argument("--config", type=str,
                    help="Path to the .yaml config file (optional, will be loaded from ckpt if not provided).")
    ap.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"],
                    help="Which data split to evaluate.")
    ap.add_argument("--decode", type=str, default="beam", choices=["greedy", "beam"], help="Decoding strategy.")
    ap.add_argument("--beam_size", type=int, default=4, help="Beam size for beam search.")
    ap.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    ap.add_argument("--device", type=str, default="auto", help="Device to use (e.g., 'cpu', 'cuda', 'auto').")
    args = ap.parse_args()

    device = resolve_device(args.device)

    # 加载配置
    if args.config:
        with open(args.config, "r", encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    else:
        # 从检查点文件中加载配置
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if "config" not in ckpt:
            raise ValueError("Config not found in checkpoint. Please provide it with --config.")
        cfg = ckpt["config"]

    # 加载模型
    model_cfg = dict(cfg["model"])
    model_cfg.pop('label_smoothing', None)  # 确保初始化时没有这个参数

    # 修复可能存在的 vocab_size 不匹配问题
    # 如果 spm 模型存在，用它的 vocab size 覆盖配置
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(cfg["data"]["spm_model_path"])
        model_cfg["vocab_size"] = sp.vocab_size()
    except Exception:
        print("Warning: Could not load SPM to verify vocab size. Using size from config.")
        pass

    model = TransformerSeq2Seq(**model_cfg).to(device)

    model_state_dict = torch.load(args.ckpt, map_location=device)["model"]
    model.load_state_dict(model_state_dict)
    print(f"Model loaded from {args.ckpt}")

    with torch.no_grad():
        eval_split(model, cfg, args.split, args.decode, args.max_new_tokens, args.beam_size, device)


if __name__ == "__main__":
    main()
