import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, use_bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)
        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        B, T, C = x.size()
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):
        B, h, T, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, h * d_k)

    def forward(self, q_inp, k_inp, v_inp, attn_mask=None):
        Q = self._split(self.W_q(q_inp))
        K = self._split(self.W_k(k_inp))
        V = self._split(self.W_v(v_inp))
        out, attn = self.attn(Q, K, V, attn_mask=attn_mask)
        out = self._merge(out)
        out = self.W_o(out)
        out = self.dropout(out)
        return out, attn


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, use_bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.ln(x + self.dropout(sublayer(x)))
