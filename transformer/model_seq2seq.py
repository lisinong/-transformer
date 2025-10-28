\
import torch
import torch.nn as nn
from .modules import MultiHeadAttention, PositionwiseFFN, ResidualLayerNorm, PositionalEncoding

BOS = 256
EOS = 257
PAD = 258

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
    def __init__(self, vocab_size=259, d_model=256, n_heads=4, d_ff=1024, n_layers=4, dropout=0.1, use_bias=True, max_len=10000):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout=dropout, use_bias=use_bias) for _ in range(n_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout=dropout, use_bias=use_bias) for _ in range(n_layers)])
        self.enc_ln = nn.LayerNorm(d_model)
        self.dec_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _causal_mask(self, T, device):
        import torch
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)[None, None, :, :]

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        device = src.device
        # encoder
        enc = self.token_emb(src); enc = self.pos_enc(enc)
        for layer in self.enc_layers:
            enc = layer(enc, src_key_padding_mask=src_key_padding_mask)
        mem = self.enc_ln(enc)
        # decoder
        dec = self.token_emb(tgt); dec = self.pos_enc(dec)
        tgt_mask = self._causal_mask(dec.size(1), device)
        for layer in self.dec_layers:
            dec = layer(dec, mem, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, mem_key_padding_mask=src_key_padding_mask)
        dec = self.dec_ln(dec)
        logits = self.lm_head(dec)
        return logits
