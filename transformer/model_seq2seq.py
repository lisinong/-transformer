import torch
import torch.nn as nn
from .modules import MultiHeadAttention, PositionwiseFFN, ResidualLayerNorm, PositionalEncoding
import math
import torch.nn.functional as F

BOS = 256
EOS = 257
PAD = 258


# ===== 新增：位置编码工厂 =====


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class LearnedPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(x.size(0), T)
        return x + self.emb(pos)


class NoPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


def build_positional_encoding(kind: str, d_model: int, max_len: int):
    kind = (kind or "sinusoidal").lower()
    if kind == "sinusoidal":
        return SinusoidalPE(d_model, max_len)
    if kind == "learned":
        return LearnedPE(d_model, max_len)
    if kind == "none":
        return NoPE(d_model, max_len)
    raise ValueError(f"Unknown positional_encoding: {kind}")


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, use_bias=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, use_bias=use_bias)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout=dropout, use_bias=use_bias)
        self.res_attn = ResidualLayerNorm(d_model, dropout=dropout)
        self.res_ffn = ResidualLayerNorm(d_model, dropout=dropout)

    def forward(self, x, src_key_padding_mask=None):
        attn_mask = None
        if src_key_padding_mask is not None:
            attn_mask = src_key_padding_mask[:, None, None, :]
        x = self.res_attn(x, lambda z: self.self_attn(z, z, z, attn_mask=attn_mask)[0])
        x = self.res_ffn(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, use_bias=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, use_bias=use_bias)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, use_bias=use_bias)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout=dropout, use_bias=use_bias)
        self.res_self = ResidualLayerNorm(d_model, dropout=dropout)
        self.res_cross = ResidualLayerNorm(d_model, dropout=dropout)
        self.res_ffn = ResidualLayerNorm(d_model, dropout=dropout)

    def forward(self, x, mem, tgt_mask=None, tgt_key_padding_mask=None, mem_key_padding_mask=None):
        self_mask = None
        if tgt_mask is not None:
            self_mask = tgt_mask
        if tgt_key_padding_mask is not None:
            pad_mask = tgt_key_padding_mask[:, None, None, :]
            self_mask = pad_mask if self_mask is None else (self_mask | pad_mask)
        x = self.res_self(x, lambda z: self.self_attn(z, z, z, attn_mask=self_mask)[0])

        cross_mask = None
        if mem_key_padding_mask is not None:
            cross_mask = mem_key_padding_mask[:, None, None, :]
        x = self.res_cross(x, lambda z: self.cross_attn(z, mem, mem, attn_mask=cross_mask)[0])

        x = self.res_ffn(x, self.ffn)
        return x


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size=259, d_model=256, n_heads=4, d_ff=1024, n_layers=4, dropout=0.1, use_bias=True,
                 positional_encoding: str = "sinusoidal",
                 max_seq_len=10000, tie_weights=False):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dropout_p = float(dropout)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        # self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        # 位置编码（可切换）
        self.pos_enc = build_positional_encoding(positional_encoding, d_model, max_seq_len)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout=dropout, use_bias=use_bias) for _ in range(n_layers)])
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout=dropout, use_bias=use_bias) for _ in range(n_layers)])
        self.enc_ln = nn.LayerNorm(d_model)
        self.dec_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight  # 权重共享（可选）

    def _causal_mask(self, T, device):
        import torch
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)[None, None, :, :]

    def encode(self, src_ids, src_key_padding_mask=None):
        """
        对源序列进行编码。
        """
        # ---- Embedding & Positional Encoding ----
        src = self.token_emb(src_ids) * (self.d_model ** 0.5)
        src = self.pos_enc(src)
        src = nn.Dropout(self.dropout_p)(src)

        # ---- Encoder ----
        enc = src
        for layer in self.enc_layers:
            enc = layer(enc, src_key_padding_mask=src_key_padding_mask)
        mem = self.enc_ln(enc)  # Encoder 的最终输出
        return mem

    # --- 【新增】: decode 方法 ---
    def decode(self, memory, tgt_ids, mem_key_padding_mask=None, tgt_key_padding_mask=None, tgt_is_causal=True):
        """
        对目标序列进行解码，使用编码器的输出 `memory`。
        """
        # ---- Embedding & Positional Encoding ----
        tgt = self.token_emb(tgt_ids) * (self.d_model ** 0.5)
        tgt = self.pos_enc(tgt)
        tgt = nn.Dropout(self.dropout_p)(tgt)

        # ---- Decoder ----
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device) if tgt_is_causal else None
        dec = tgt
        for layer in self.dec_layers:
            dec = layer(
                dec,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                mem_key_padding_mask=mem_key_padding_mask,
            )
        dec = self.dec_ln(dec)

        # ---- LM Head ----
        logits = self.lm_head(dec)
        return logits

        # --- 【修改】: forward 方法现在调用 encode 和 decode ---

    def forward(self, src_ids, tgt_ids, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        完整的 Teacher-Forcing 前向传播。
        """
        memory = self.encode(src_ids, src_key_padding_mask)
        logits = self.decode(memory, tgt_ids, mem_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask)
        return logits

    @torch.no_grad()  # 确保在生成时不计算梯度
    def generate(self, src, src_key_padding_mask, max_new_tokens, bos, eos, pad):
        """
        实现贪心解码 (Greedy Decoding) 的方法。

        Args:
            src (torch.Tensor): 源序列，形状为 [N, S_len]
            src_key_padding_mask (torch.Tensor): 源序列的padding mask，形状为 [N, S_len]
            max_new_tokens (int): 最大生成 token 数量
            bos (int): 开始符 (BOS) 的 ID
            eos (int): 结束符 (EOS) 的 ID
            pad (int): 填充符 (PAD) 的 ID

        Returns:
            torch.Tensor: 生成的序列，形状为 [N, T_len]
        """
        device = src.device
        batch_size = src.size(0)

        # 1. 对输入进行编码，这个操作只需要一次
        encoder_output = self.encode(src, src_key_padding_mask)

        # 2. 初始化解码器的输入，所有序列都以 BOS 开始
        # 形状为 [N, 1]
        tgt = torch.full((batch_size, 1), bos, dtype=torch.long, device=device)

        # 3. 用于跟踪哪些序列已经生成了 EOS
        # 初始时，所有序列都未完成
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        min_len = 5
        no_repeat_ngram_size = 3  # ★ 关键：不允许重复 3-gram
        repetition_penalty = 1.2  # ★ 关键：已出现过的 token 降权
        # 4. 自回归生成循环
        for _ in range(max_new_tokens):
            # 解码一步
            logits = self.decode(encoder_output, tgt, src_key_padding_mask, None, True)

            # 只取最后一个时间步的 logits 进行预测
            # 形状为 [N, vocab_size]
            next_token_logits = logits[:, -1, :]
            next_token_logits[:, pad] = float("-inf")
            # 从第二步开始禁止再次选 BOS（第一步必须是 BOS）
            if tgt.size(1) > 1:
                next_token_logits[:, bos] = float("-inf")
            if _ < min_len:
                next_token_logits[:, eos] = float("-inf")
            # ★ 关键：不允许重复 n-gram
            with torch.no_grad():
                for i in range(batch_size):
                    if finished[i]:
                        continue
                    prev_tokens = tgt[i].tolist()
                    for tok in set(prev_tokens):
                        if tok in (bos, eos, pad):
                            continue
                        # 将该 token 的logit 降权（等价于概率^1/penalty）
                        next_token_logits[i, tok] /= repetition_penalty
            if no_repeat_ngram_size > 0:
                n = no_repeat_ngram_size
                with torch.no_grad():
                    for i in range(batch_size):
                        if finished[i] or tgt.size(1) < n - 1:
                            continue
                        # 构造已经出现过的 n-gram 前缀 -> 后继集合
                        history = tgt[i].tolist()
                        banned = set()
                        # 建表： (w_{t-n+1}, ..., w_{t-1}) -> {w_t}
                        prefix2next = {}
                        for j in range(len(history) - n + 1):
                            prefix = tuple(history[j:j + n - 1])
                            nxt = history[j + n - 1]
                            prefix2next.setdefault(prefix, set()).add(nxt)
                        # 取当前上下文前缀，ban 所有已见过的 next
                        cur_prefix = tuple(history[-(n - 1):])
                        for nxt in prefix2next.get(cur_prefix, []):
                            next_token_logits[i, nxt] = float("-inf")
            # 如能拿到 UNK 的 id（比如 sp.unk_id()），也可以屏蔽：
            # next_token_logits[:, unk] = float("-inf")
            # 5. 贪心策略：选择概率最高的 token
            # 形状为 [N, 1]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            # 6. 将新生成的 token 拼接到现有序列后面
            tgt = torch.cat([tgt, next_token], dim=1)

            # 7. 更新已完成的序列状态
            # 如果新生成的 token 是 EOS，则将对应序列标记为完成
            finished = finished | (next_token.squeeze() == eos)

            # 8. 如果一个批次中的所有序列都已完成，则提前退出循环
            if finished.all():
                break

        return tgt
