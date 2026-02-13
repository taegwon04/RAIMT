"""
RAILT: Recurrent Memory AttentIon for Large-Scale Interaction Knowledge Tracing

Paper-aligned implementation.
Each module is annotated with the corresponding equation numbers
from the paper for clarity.

Architecture Overview:
    Encoder: Problem ID + Skill → RMA Block → Gated Memory Update
    Decoder: Response → RMA Block → Cross-Attention → Gated Memory Update
    Output:  LayerNorm → Linear → Sigmoid
"""

import torch
import torch.nn as nn


class RMABlock(nn.Module):
    """
    Recurrent Memory Attention Block (Eq. 2-4)
    [mem ; chunk] + P → MHA → FFN → split → [mem_candidate ; h]
    """
    def __init__(self, d_model, n_heads, dropout, chunk_len, mem_len):
        super().__init__()
        self.mem_len = mem_len

        self.pos_emb = nn.Embedding(mem_len + chunk_len, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
        )
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, chunk, mem):
        device = chunk.device
        L_c = chunk.size(1)

        z = torch.cat([mem, chunk], dim=1)
        z = z + self.pos_emb(torch.arange(z.size(1), device=device).unsqueeze(0))

        T = z.size(1)
        mask = torch.zeros(T, T, dtype=torch.bool, device=device)
        mask[self.mem_len:, self.mem_len:] = torch.triu(
            torch.ones(L_c, L_c, dtype=torch.bool, device=device), diagonal=1)

        z_norm = self.ln_attn(z)
        attn_out, _ = self.self_attn(z_norm, z_norm, z_norm, attn_mask=mask)
        z = z + self.dropout(attn_out)
        z = z + self.dropout(self.ffn(self.ln_ffn(z)))

        return z[:, self.mem_len:, :], z[:, :self.mem_len, :]   # h_k, mem_candidate


class GatedMemoryUpdate(nn.Module):
    """Gated Memory Update (Eq. 5)"""
    def __init__(self, d_model):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, mem_cand, mem_prev):
        g = torch.sigmoid(self.gate_proj(mem_cand))
        return g * mem_cand + (1 - g) * mem_prev


class RAILTEncoder(nn.Module):
    """Encoder: Problem ID + Skill → RMA → Gated Memory Update (Eq. 6)"""
    def __init__(self, d_model, n_heads, total_prob, total_skill, dropout, chunk_len, mem_len):
        super().__init__()
        self.chunk_len = chunk_len

        self.emb_prob = nn.Embedding(total_prob, d_model, padding_idx=0)
        self.emb_skill = nn.Embedding(total_skill, d_model, padding_idx=0)
        self.mem_init = nn.Parameter(torch.zeros(1, mem_len, d_model))
        nn.init.xavier_uniform_(self.mem_init)

        self.rma = RMABlock(d_model, n_heads, dropout, chunk_len, mem_len)
        self.gate = GatedMemoryUpdate(d_model)

    def forward(self, in_prob, in_skill, first_block=True):
        if first_block:
            x = self.emb_prob(in_prob) + self.emb_skill(in_skill)
            L = in_prob.size(1)
        else:
            x = in_prob
            L = x.size(1)

        mem = self.mem_init.expand(x.size(0), -1, -1)
        chunks_out = []

        for start in range(0, L, self.chunk_len):
            chunk = x[:, start:min(start + self.chunk_len, L), :]
            if chunk.size(1) == 0:
                continue
            h_k, mem_cand = self.rma(chunk, mem)
            mem = self.gate(mem_cand, mem) if start > 0 else mem_cand
            chunks_out.append(h_k)

        return torch.cat(chunks_out, dim=1), mem


class RAILTDecoder(nn.Module):
    """Decoder: Response → RMA → Cross-Attention → Gated Memory Update (Eq. 7-8)"""
    def __init__(self, d_model, total_res, n_heads, dropout, chunk_len, mem_len):
        super().__init__()
        self.chunk_len = chunk_len

        self.emb_res = nn.Embedding(total_res, d_model, padding_idx=0)
        self.mem_init = nn.Parameter(torch.zeros(1, mem_len, d_model))
        nn.init.xavier_uniform_(self.mem_init)
        self.mem_proj = nn.Linear(d_model, d_model)

        self.rma = RMABlock(d_model, n_heads, dropout, chunk_len, mem_len)

        self.enc_mem_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_cross_q = nn.LayerNorm(d_model)
        self.ln_cross_kv = nn.LayerNorm(d_model)

        self.ffn_cross = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
        )
        self.ln_ffn_cross = nn.LayerNorm(d_model)
        self.gate = GatedMemoryUpdate(d_model)
        self.dropout = nn.Dropout(dropout)

    def _cross_attention(self, h_k, enc_chunk, enc_mem):
        device = h_k.device
        L_c = h_k.size(1)
        query = self.ln_cross_q(h_k)

        if enc_mem is not None:
            proj_mem = self.enc_mem_proj(enc_mem)
            kv = self.ln_cross_kv(torch.cat([proj_mem, enc_chunk], dim=1))
            M = proj_mem.size(1)
            mask = torch.zeros(L_c, M + L_c, dtype=torch.bool, device=device)
            mask[:, M:] = torch.triu(torch.ones(L_c, L_c, dtype=torch.bool, device=device), diagonal=1)
        else:
            kv = self.ln_cross_kv(enc_chunk)
            mask = torch.triu(torch.ones(L_c, L_c, dtype=torch.bool, device=device), diagonal=1)

        cross_out, _ = self.cross_attn(query, kv, kv, attn_mask=mask)
        h_k = h_k + self.dropout(cross_out)
        h_k = h_k + self.dropout(self.ffn_cross(self.ln_ffn_cross(h_k)))
        return h_k

    def forward(self, in_res, enc_out, enc_mem, first_block=True):
        if first_block:
            x = self.emb_res(in_res)
            L = in_res.size(1)
        else:
            x = in_res
            L = x.size(1)

        mem = self.mem_proj(enc_mem) if enc_mem is not None else self.mem_init.expand(x.size(0), -1, -1)
        chunks_out = []

        for start in range(0, L, self.chunk_len):
            end = min(start + self.chunk_len, L)
            chunk = x[:, start:end, :]
            enc_chunk = enc_out[:, start:end, :]
            if chunk.size(1) == 0:
                continue

            h_k, mem_cand = self.rma(chunk, mem)
            h_k = self._cross_attention(h_k, enc_chunk, enc_mem)
            mem = self.gate(mem_cand, mem) if start > 0 else mem_cand
            chunks_out.append(h_k)

        return torch.cat(chunks_out, dim=1)


class RAILT(nn.Module):
    def __init__(self, d_model, num_en, num_de, heads_en, heads_de,
                 total_prob, total_skill, total_res, dropout,
                 chunk_len=50, mem_len=16):
        super().__init__()
        self.num_en = num_en
        self.num_de = num_de

        self.encoders = nn.ModuleList([
            RAILTEncoder(d_model, heads_en, total_prob, total_skill, dropout, chunk_len, mem_len)
            for _ in range(num_en)
        ])
        self.decoders = nn.ModuleList([
            RAILTDecoder(d_model, total_res, heads_de, dropout, chunk_len, mem_len)
            for _ in range(num_de)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, in_prob, in_skill, in_res):
        # Encoder
        first_block = True
        enc_out, enc_mem = None, None
        for x in range(self.num_en):
            if x >= 1:
                first_block = False
            enc_out, enc_mem = self.encoders[x](
                enc_out if not first_block else in_prob,
                None if not first_block else in_skill,
                first_block=first_block)

        # Decoder
        first_block = True
        dec_out = None
        for x in range(self.num_de):
            if x >= 1:
                first_block = False
            dec_out = self.decoders[x](
                dec_out if not first_block else in_res,
                enc_out,
                enc_mem if first_block else None,
                first_block=first_block)

        # Eq. (9)
        return torch.sigmoid(self.out(self.final_ln(dec_out)))


## forward prop on dummy data

seq_len = 100
total_prob = 1200
total_skill = 234
total_res = 2


def random_data(bs, seq_len, total_prob, total_skill, total_res=2):
    prob = torch.randint(0, total_prob, (bs, seq_len))
    skill = torch.randint(0, total_skill, (bs, seq_len))
    res = torch.randint(0, total_res, (bs, seq_len))
    return prob, skill, res


in_prob, in_skill, in_res = random_data(64, seq_len, total_prob, total_skill, total_res)

model = RAILT(d_model=128,
              num_en=6,
              num_de=6,
              heads_en=8,
              heads_de=8,
              total_prob=total_prob,
              total_skill=total_skill,
              total_res=total_res,
              dropout=0.1,
              chunk_len=50,
              mem_len=16)

outs = model(in_prob, in_skill, in_res)

print(outs.shape)
