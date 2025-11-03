import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
import sentencepiece as spm

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


def beam_search(model, src, src_kpm, bos, eos, pad, max_new_tokens, beam_size, device="cpu", lp_alpha=0.6):
    """
    为单个样本执行 Beam Search。
    注意：此实现仅支持 batch_size=1 的输入。
    """
    assert src.size(0) == 1, "Beam search implementation only supports batch_size=1"

    beams = [(torch.tensor([[bos]], dtype=torch.long, device=device), 0.0)]  # (sequence, log_probability_score)

    with torch.no_grad():
        encoder_out = model.encode(src, src_key_padding_mask=src_kpm)

        for _ in range(max_new_tokens):
            new_beams = []

            # 检查是否有 beam 已经生成了 EOS
            has_ended_beams = False
            for seq, score in beams:
                if seq[0, -1].item() == eos:
                    # 这个 beam 已经完成，直接加入下一轮候选，不再扩展
                    new_beams.append((seq, score))
                    has_ended_beams = True

            if has_ended_beams and len(new_beams) == beam_size:
                # 如果所有的 beam 都已结束，提前终止
                if all(b[0][0, -1].item() == eos for b in new_beams):
                    break

            for seq, score in beams:
                if seq[0, -1].item() == eos:
                    continue

                # 解码
                logits = model.decode(encoder_out, seq, None, None)
                logp = F.log_softmax(logits[:, -1, :], dim=-1)  # 只看最后一个 token 的 log probabilities
                logp[:, pad] = -float("inf")
                if seq.size(1) > 1:
                    logp[:, bos] = -float("inf")
                # 获取 top-k 候选项
                topk = torch.topk(logp, beam_size, dim=-1)

                for k in range(beam_size):
                    next_token_id = topk.indices[0, k].view(1, 1)
                    next_token_logp = float(topk.values[0, k])

                    new_seq = torch.cat([seq, next_token_id], dim=1)
                    new_score = score + next_token_logp
                    new_beams.append((new_seq, new_score))

            # 根据分数排序并剪枝
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

            # 检查是否所有 beam 都已结束
            if all(b[0][0, -1].item() == eos for b in beams):
                break

    # 长度惩罚并选择最佳 beam
    # (score / length^alpha)
    best_beam = max(beams, key=lambda x: x[1] / ((5 + x[0].size(1)) / 6) ** lp_alpha)
    return best_beam[0]  # 只返回最佳序列


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
        hyps_text = sp.decode(ys.tolist())
        refs_text = sp.decode(tout.tolist())

        all_hyps.extend(hyps_text)
        all_refs.extend(refs_text)

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
