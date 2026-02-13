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
    """Recurrent Memory Attention Block (Section 3.1, Eq. 2–4)"""

    def __init__(self, d_model, n_heads, dropout, chunk_len, mem_len):
        super().__init__()
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        self.pos_emb = nn.Embedding(mem_len + chunk_len, d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
        )
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, chunk, mem, return_attn=False):
        device = chunk.device
        L_c = chunk.size(1)

        # Eq. (2): z^(k) = [mem^(k-1) ; chunk^(k)] + P
        z = torch.cat([mem, chunk], dim=1)
        z = z + self.pos_emb(torch.arange(z.size(1), device=device).unsqueeze(0))

        # Causal mask: memory=global, chunk=causal
        T = z.size(1)
        mask = torch.zeros(T, T, dtype=torch.bool, device=device)
        mask[self.mem_len:, self.mem_len:] = torch.triu(
            torch.ones(L_c, L_c, dtype=torch.bool, device=device), diagonal=1
        )

        # Eq. (3): ẑ = z + MHA(LN(z)),  RMA(z) = ẑ + FFN(LN(ẑ))
        z_norm = self.ln_attn(z)
        attn_out, attn_w = self.self_attn(
            z_norm, z_norm, z_norm, attn_mask=mask,
            need_weights=return_attn, average_attn_weights=False,
        )
        z = z + self.dropout(attn_out)
        z = z + self.dropout(self.ffn(self.ln_ffn(z)))

        # Eq. (4): split → [mem_candidate ; h^(k)]
        mem_cand = z[:, :self.mem_len, :]
        h_k = z[:, self.mem_len:, :]

        if return_attn:
            return h_k, mem_cand, attn_w
        return h_k, mem_cand


class GatedMemoryUpdate(nn.Module):
    """Gated Memory Update (Eq. 5)"""

    def __init__(self, d_model):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, mem_cand, mem_prev):
        g = torch.sigmoid(self.gate_proj(mem_cand))
        return g * mem_cand + (1 - g) * mem_prev


class RAILTEncoder(nn.Module):
    """Encoder (Section 3.2, Eq. 6)"""

    def __init__(self, d_model, n_heads, total_prob, total_skill, dropout,
                 chunk_len=50, mem_len=16):
        super().__init__()
        self.chunk_len = chunk_len

        self.emb_problem = nn.Embedding(total_prob, d_model, padding_idx=0)
        self.emb_skill = nn.Embedding(total_skill, d_model, padding_idx=0)
        self.mem_init = nn.Parameter(torch.zeros(1, mem_len, d_model))
        nn.init.xavier_uniform_(self.mem_init)

        self.rma = RMABlock(d_model, n_heads, dropout, chunk_len, mem_len)
        self.gate = GatedMemoryUpdate(d_model)

    def forward(self, in_prob, in_skill, first_block=True, return_attn=False):
        if first_block:
            x = self.emb_problem(in_prob) + self.emb_skill(in_skill)  # Eq. (6)
            L = in_prob.size(1)
        else:
            x = in_prob
            L = x.size(1)

        mem = self.mem_init.expand(x.size(0), -1, -1)
        chunks_out, attn_maps = [], []

        for start in range(0, L, self.chunk_len):
            chunk = x[:, start:min(start + self.chunk_len, L), :]
            if chunk.size(1) == 0:
                continue

            if return_attn:
                h_k, mem_cand, w = self.rma(chunk, mem, return_attn=True)
                attn_maps.append(w.detach().cpu())
            else:
                h_k, mem_cand = self.rma(chunk, mem)

            mem = self.gate(mem_cand, mem) if start > 0 else mem_cand  # Eq. (5)
            chunks_out.append(h_k)

        enc_out = torch.cat(chunks_out, dim=1) if chunks_out else x
        if return_attn:
            return enc_out, mem, attn_maps
        return enc_out, mem


class RAILTDecoder(nn.Module):
    """Decoder (Section 3.3, Eq. 7–8)"""

    def __init__(self, d_model, total_res, n_heads, dropout,
                 chunk_len=50, mem_len=16):
        super().__init__()
        self.chunk_len = chunk_len

        self.emb_response = nn.Embedding(total_res, d_model, padding_idx=0)
        self.mem_init = nn.Parameter(torch.zeros(1, mem_len, d_model))
        nn.init.xavier_uniform_(self.mem_init)
        self.mem_proj = nn.Linear(d_model, d_model)

        self.rma = RMABlock(d_model, n_heads, dropout, chunk_len, mem_len)

        self.enc_mem_proj = nn.Linear(d_model, d_model)  # W_m in Eq. (7)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_cross_q = nn.LayerNorm(d_model)
        self.ln_cross_kv = nn.LayerNorm(d_model)

        self.ffn_cross = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
        )
        self.ln_ffn_cross = nn.LayerNorm(d_model)
        self.gate = GatedMemoryUpdate(d_model)
        self.dropout = nn.Dropout(dropout)

    def _cross_attention(self, h_k, enc_chunk, enc_mem, return_attn=False):
        device = h_k.device
        L_c = h_k.size(1)
        query = self.ln_cross_q(h_k)

        if enc_mem is not None:
            # Eq. (7): kv = [W_m · mem_enc ; h_enc^(k)]
            proj_mem = self.enc_mem_proj(enc_mem)
            kv = self.ln_cross_kv(torch.cat([proj_mem, enc_chunk], dim=1))
            M = proj_mem.size(1)
            mask = torch.zeros(L_c, M + L_c, dtype=torch.bool, device=device)
            mask[:, M:] = torch.triu(
                torch.ones(L_c, L_c, dtype=torch.bool, device=device), diagonal=1
            )
        else:
            kv = self.ln_cross_kv(enc_chunk)
            mask = torch.triu(
                torch.ones(L_c, L_c, dtype=torch.bool, device=device), diagonal=1
            )

        # Eq. (8): h_cross = MHA(h^(k), kv, kv)
        cross_out, cross_w = self.cross_attn(
            query, kv, kv, attn_mask=mask,
            need_weights=return_attn, average_attn_weights=False,
        )
        h_k = h_k + self.dropout(cross_out)
        h_k = h_k + self.dropout(self.ffn_cross(self.ln_ffn_cross(h_k)))

        if return_attn:
            return h_k, cross_w
        return h_k

    def forward(self, in_res, enc_out, enc_mem,
                first_block=True, return_attn=False):
        if first_block:
            x = self.emb_response(in_res)
            L = in_res.size(1)
        else:
            x = in_res
            L = x.size(1)

        mem = (self.mem_proj(enc_mem) if enc_mem is not None
               else self.mem_init.expand(x.size(0), -1, -1))

        chunks_out, self_attn_maps, cross_attn_maps = [], [], []

        for start in range(0, L, self.chunk_len):
            chunk = x[:, start:min(start + self.chunk_len, L), :]
            enc_chunk = enc_out[:, start:min(start + self.chunk_len, L), :]
            if chunk.size(1) == 0:
                continue

            if return_attn:
                h_k, mem_cand, sw = self.rma(chunk, mem, return_attn=True)
                self_attn_maps.append(sw.detach().cpu())
                h_k, cw = self._cross_attention(h_k, enc_chunk, enc_mem, True)
                cross_attn_maps.append(cw.detach().cpu())
            else:
                h_k, mem_cand = self.rma(chunk, mem)
                h_k = self._cross_attention(h_k, enc_chunk, enc_mem)

            mem = self.gate(mem_cand, mem) if start > 0 else mem_cand  # Eq. (5)
            chunks_out.append(h_k)

        dec_out = torch.cat(chunks_out, dim=1) if chunks_out else x
        if return_attn:
            return dec_out, self_attn_maps, cross_attn_maps
        return dec_out


class RAILT(nn.Module):
    """RAILT (Eq. 9)"""

    def __init__(self, d_model, num_enc_layers, num_dec_layers,
                 n_heads_enc, n_heads_dec, total_prob, total_skill, total_res,
                 dropout, chunk_len=50, mem_len=16):
        super().__init__()
        self.encoders = nn.ModuleList([
            RAILTEncoder(d_model, n_heads_enc, total_prob, total_skill,
                         dropout, chunk_len, mem_len)
            for _ in range(num_enc_layers)
        ])
        self.decoders = nn.ModuleList([
            RAILTDecoder(d_model, total_res, n_heads_dec,
                         dropout, chunk_len, mem_len)
            for _ in range(num_dec_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, in_prob, in_skill, in_res, return_attn=False):
        enc_out, enc_mem, enc_attns = None, None, []

        for i, enc in enumerate(self.encoders):
            first = (i == 0)
            if first and return_attn:
                enc_out, enc_mem, enc_attns = enc(
                    in_prob, in_skill, first_block=True, return_attn=True)
            else:
                enc_out, enc_mem = enc(
                    enc_out if not first else in_prob,
                    None if not first else in_skill, first_block=first)

        dec_out, dec_sa, dec_ca = None, [], []

        for i, dec in enumerate(self.decoders):
            first = (i == 0)
            if first and return_attn:
                dec_out, dec_sa, dec_ca = dec(
                    in_res, enc_out, enc_mem,
                    first_block=True, return_attn=True)
            else:
                dec_out = dec(
                    dec_out if not first else in_res, enc_out,
                    enc_mem if first else None, first_block=first)

        # Eq. (9): ŷ_t = σ(W_o · LN(h_t) + b_o)
        pred = torch.sigmoid(self.out_proj(self.final_ln(dec_out))).squeeze(-1)

        if return_attn:
            return pred, enc_attns, dec_sa, dec_ca
        return pred
