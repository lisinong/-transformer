\
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in Vaswani et al. (2017).
    Produces a [1, T, d_model] buffer added to token embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T, :]

class ScaledDotProductAttention(nn.Module):
    """
    Computes attention weights and output: softmax(QK^T / sqrt(d_k)) V
    Optional attn_mask for causal masking (True/1 indicates to mask out).
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask=None):
        # Q,K,V: [B, h, T, d_k]
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B,h,T,T]
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)  # [B,h,T,T]
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)       # [B,h,T,d_k]
        return out, attn

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention (self: Q=K=V from same input).
    Implements projections with separate linear layers and an output projection.
    """
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

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        # Project and reshape for heads
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # [B,h,T,d_k]
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Apply attention
        out, attn = self.attn(Q, K, V, attn_mask=attn_mask)  # out: [B,h,T,d_k]
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B,T,d_model]
        out = self.W_o(out)                                   # [B,T,d_model]
        out = self.dropout(out)
        return out, attn

class PositionwiseFFN(nn.Module):
    """
    Position-wise feed-forward network: FFN(x) = W2 * GELU(W1 * x) with dropout.
    """
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
    """
    Pre-LN residual block wrapper:
    y = x + Dropout(sublayer(LayerNorm(x)))
    """
    def __init__(self, d_model: int, dropout: float = 0.0, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.ln(x)))
