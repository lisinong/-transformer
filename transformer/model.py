import torch
import torch.nn as nn
from .modules import MultiHeadSelfAttention, PositionwiseFFN, ResidualLayerNorm, PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, use_bias=True):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout, use_bias=use_bias)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout=dropout, use_bias=use_bias)
        self.res_attn = ResidualLayerNorm(d_model, dropout=dropout)
        self.res_ffn = ResidualLayerNorm(d_model, dropout=dropout)

    def forward(self, x, attn_mask=None):
        x = self.res_attn(x, lambda z: self.mhsa(z, attn_mask=attn_mask)[0])
        x = self.res_ffn(x, self.ffn)
        return x


class TransformerEncoderLM(nn.Module):
    """
    Encoder-only Transformer for causal language modeling over byte tokens.
    """

    def __init__(self, vocab_size=256, d_model=256, n_heads=4, d_ff=1024, n_layers=4, dropout=0.1, use_bias=True,
                 max_len=10000):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout=dropout, use_bias=use_bias)
            for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _causal_mask(self, T, device):
        # True where masked (upper triangle)
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, idx):
        # idx: [B, T] integer byte tokens
        B, T = idx.size()
        device = idx.device
        x = self.token_emb(idx)  # [B,T,C]
        x = self.pos_enc(x)  # add sinusoidal positions
        attn_mask = self._causal_mask(T, device)[None, None, :, :]  # [1,1,T,T] broadcast to [B,h,T,T]
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]
        return logits
