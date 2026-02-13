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

# ---------------------------------------------------------------------------
# Recurrent Memory Attention Block  (Section 3.1, Eq. 2–4)
# ---------------------------------------------------------------------------
class RMABlock(nn.Module):
    """
    Recurrent Memory Attention Block.

    At each chunk step k:
        z^(k) = [mem^(k-1) ; chunk^(k)] + P            ... Eq. (2)
        ẑ     = z^(k) + MHA(LN(z^(k)))                  ... Eq. (3)
        RMA(z^(k)) = ẑ + FFN(LN(ẑ))                     ... Eq. (3)
        [mem_cand^(k) ; h^(k)] = RMA(z^(k))             ... Eq. (4)

    Memory tokens have global attention access; chunk tokens are
    restricted by a causal mask to prevent attending to future positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 chunk_len: int, mem_len: int):
        super().__init__()
        self.d_model = d_model
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        # Learnable positional embedding P  (Eq. 2)
        self.pos_emb = nn.Embedding(mem_len + chunk_len, d_model)

        # Multi-Head Self-Attention  (Eq. 3)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Position-wise Feed-Forward Network  (Eq. 3)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        # Pre-Layer Normalization
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, chunk, mem, return_attn=False):
        """
        Args:
            chunk: (B, L_c, D)  — current chunk tokens
            mem:   (B, M, D)    — previous memory state
        Returns:
            h_k:          (B, L_c, D) — chunk representation
            mem_candidate: (B, M, D)  — new memory candidate (before gating)
            attn_weights:  (optional)  — attention weights for visualization
        """
        device = chunk.device
        L_c = chunk.size(1)

        # --- Eq. (2): z^(k) = [mem^(k-1) ; chunk^(k)] + P ---
        z = torch.cat([mem, chunk], dim=1)
        pos_ids = torch.arange(z.size(1), device=device).unsqueeze(0)
        z = z + self.pos_emb(pos_ids)

        # --- Causal mask: memory has global access; chunk is causal ---
        total_len = z.size(1)
        causal_mask = torch.zeros(
            total_len, total_len, dtype=torch.bool, device=device
        )
        causal_mask[self.mem_len:, self.mem_len:] = torch.triu(
            torch.ones(L_c, L_c, dtype=torch.bool, device=device), diagonal=1
        )

        # --- Eq. (3): ẑ = z^(k) + MHA(LN(z^(k))) ---
        z_norm = self.ln_attn(z)
        attn_out, attn_weights = self.self_attn(
            z_norm, z_norm, z_norm,
            attn_mask=causal_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        z = z + self.dropout(attn_out)

        # --- Eq. (3): RMA(z^(k)) = ẑ + FFN(LN(ẑ)) ---
        z = z + self.dropout(self.ffn(self.ln_ffn(z)))

        # --- Eq. (4): split → [mem_candidate^(k) ; h^(k)] ---
        mem_candidate = z[:, :self.mem_len, :]
        h_k = z[:, self.mem_len:, :]

        if return_attn:
            return h_k, mem_candidate, attn_weights
        return h_k, mem_candidate


# ---------------------------------------------------------------------------
# Gated Memory Update  (Section 3.1, Eq. 5)
# ---------------------------------------------------------------------------
class GatedMemoryUpdate(nn.Module):
    """
    Gated memory update mechanism.

        G^(k) = σ(W_g · mem_cand^(k))                   ... Eq. (5)
        mem^(k) = G^(k) ⊙ mem_cand^(k)
                  + (1 - G^(k)) ⊙ mem^(k-1)             ... Eq. (5)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model)     # W_g

    def forward(self, mem_candidate, mem_prev):
        gate = torch.sigmoid(self.gate_proj(mem_candidate))  # G^(k)
        return gate * mem_candidate + (1 - gate) * mem_prev  # mem^(k)


# ---------------------------------------------------------------------------
# RAILT Encoder  (Section 3.2, Eq. 6)
# ---------------------------------------------------------------------------
class RAILTEncoder(nn.Module):
    """
    Encoder: embeds problem & skill IDs, then processes chunks
    through an RMA block with gated memory propagation.

        x_t = Emb_prob(q_t) + Emb_skill(s_t)            ... Eq. (6)

    Outputs per-timestep representations h and the final memory
    mem^(K), which serves as a compressed summary of the input.
    """

    def __init__(self, d_model: int, n_heads: int, total_ex: int,
                 total_cat: int, dropout: float,
                 chunk_len: int = 50, mem_len: int = 16):
        super().__init__()
        self.d_model = d_model
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        # --- Eq. (6): input embeddings ---
        self.emb_problem = nn.Embedding(total_ex, d_model, padding_idx=0)
        self.emb_skill = nn.Embedding(total_cat, d_model, padding_idx=0)

        # Learnable initial memory mem^(0)
        self.mem_init = nn.Parameter(torch.zeros(1, mem_len, d_model))
        nn.init.xavier_uniform_(self.mem_init)

        # RMA block & gated update
        self.rma = RMABlock(d_model, n_heads, dropout, chunk_len, mem_len)
        self.gate = GatedMemoryUpdate(d_model)

    def forward(self, in_ex, in_cat, first_block=True, return_attn=False):
        """
        Args:
            in_ex:  problem IDs (B, L) or hidden states (B, L, D)
            in_cat: skill IDs   (B, L) — ignored when first_block=False
        """
        if first_block:
            # --- Eq. (6) ---
            x = self.emb_problem(in_ex) + self.emb_skill(in_cat)
            B, L = in_ex.shape
        else:
            x = in_ex
            B, L, _ = x.shape

        mem = self.mem_init.expand(B, -1, -1)
        chunks_out, attn_maps = [], []

        for start in range(0, L, self.chunk_len):
            chunk = x[:, start:min(start + self.chunk_len, L), :]
            if chunk.size(1) == 0:
                continue

            # --- RMA block (Eq. 2–4) ---
            if return_attn:
                h_k, mem_candidate, weights = self.rma(
                    chunk, mem, return_attn=True
                )
                attn_maps.append(weights.detach().cpu())
            else:
                h_k, mem_candidate = self.rma(chunk, mem)

            # --- Gated memory update (Eq. 5) ---
            # Applied immediately after RMA in encoder (Section 3.1)
            mem = self.gate(mem_candidate, mem) if start > 0 else mem_candidate
            chunks_out.append(h_k)

        enc_out = torch.cat(chunks_out, dim=1) if chunks_out else x

        if return_attn:
            return enc_out, mem, attn_maps
        return enc_out, mem


# ---------------------------------------------------------------------------
# RAILT Decoder  (Section 3.3, Eq. 7–9)
# ---------------------------------------------------------------------------
class RAILTDecoder(nn.Module):
    """
    Decoder: embeds response sequence, applies RMA self-attention,
    then cross-attends to encoder outputs before gated memory update.

    Cross-attention key/value construction:
        kv^(k) = [W_m · mem_enc ; h^(k)_enc]            ... Eq. (7)

    Cross-attention:
        h^(k)_cross = MHA(h^(k), kv^(k), kv^(k))       ... Eq. (8)

    Note: Decoder memory is excluded from the cross-attention query
    to prevent information leakage (Section 3.3).

    Memory update is applied *after* cross-attention (Section 3.1).
    """

    def __init__(self, d_model: int, total_res: int, n_heads: int,
                 dropout: float, chunk_len: int = 50, mem_len: int = 16):
        super().__init__()
        self.d_model = d_model
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        # Response embedding
        self.emb_response = nn.Embedding(total_res, d_model, padding_idx=0)

        # Learnable initial decoder memory
        self.mem_init = nn.Parameter(torch.zeros(1, mem_len, d_model))
        nn.init.xavier_uniform_(self.mem_init)

        # Projection: encoder memory → decoder initial memory
        self.mem_proj = nn.Linear(d_model, d_model)

        # RMA block (self-attention part)
        self.rma = RMABlock(d_model, n_heads, dropout, chunk_len, mem_len)

        # --- Eq. (7): encoder memory projection W_m ---
        self.enc_mem_proj = nn.Linear(d_model, d_model)

        # --- Eq. (8): cross-attention ---
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_cross_q = nn.LayerNorm(d_model)
        self.ln_cross_kv = nn.LayerNorm(d_model)

        # Post cross-attention FFN
        self.ffn_cross = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_ffn_cross = nn.LayerNorm(d_model)

        # Gated memory update
        self.gate = GatedMemoryUpdate(d_model)
        self.dropout = nn.Dropout(dropout)

    def _cross_attention(self, h_k, enc_chunk, enc_mem, return_attn=False):
        """
        Recurrent Memory Cross-Attention (Eq. 7–8).

        Query:     h^(k) (decoder chunk representation only)
        Key/Value: [W_m · mem_enc ; h^(k)_enc]
        """
        device = h_k.device
        L_c = h_k.size(1)
        query = self.ln_cross_q(h_k)

        if enc_mem is not None:
            # --- Eq. (7): kv^(k) = [W_m · mem_enc ; h^(k)_enc] ---
            proj_mem = self.enc_mem_proj(enc_mem)
            kv = torch.cat([proj_mem, enc_chunk], dim=1)
            kv = self.ln_cross_kv(kv)

            M = proj_mem.size(1)
            cross_mask = torch.zeros(
                L_c, M + L_c, dtype=torch.bool, device=device
            )
            cross_mask[:, M:] = torch.triu(
                torch.ones(L_c, L_c, dtype=torch.bool, device=device),
                diagonal=1,
            )
        else:
            kv = self.ln_cross_kv(enc_chunk)
            cross_mask = torch.triu(
                torch.ones(L_c, L_c, dtype=torch.bool, device=device),
                diagonal=1,
            )

        # --- Eq. (8): h^(k)_cross = MHA(h^(k), kv, kv) ---
        cross_out, cross_weights = self.cross_attn(
            query, kv, kv,
            attn_mask=cross_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        h_k = h_k + self.dropout(cross_out)

        # Post cross-attention FFN
        h_k = h_k + self.dropout(self.ffn_cross(self.ln_ffn_cross(h_k)))

        if return_attn:
            return h_k, cross_weights
        return h_k

    def forward(self, in_res, enc_out, enc_mem,
                first_block=True, return_attn=False):
        """
        Args:
            in_res:  response IDs (B, L) or hidden states (B, L, D)
            enc_out: encoder per-timestep output (B, L, D)
            enc_mem: encoder final memory (B, M, D)
        """
        if first_block:
            x = self.emb_response(in_res)
            B, L = in_res.shape
        else:
            x = in_res
            B, L, _ = x.shape

        # Initialize decoder memory from encoder memory
        mem = (self.mem_proj(enc_mem) if enc_mem is not None
               else self.mem_init.expand(B, -1, -1))

        chunks_out = []
        self_attn_maps, cross_attn_maps = [], []

        for start in range(0, L, self.chunk_len):
            end = min(start + self.chunk_len, L)
            chunk = x[:, start:end, :]
            enc_chunk = enc_out[:, start:end, :]
            if chunk.size(1) == 0:
                continue

            # --- RMA self-attention (Eq. 2–4) ---
            if return_attn:
                h_k, mem_candidate, self_w = self.rma(
                    chunk, mem, return_attn=True
                )
                self_attn_maps.append(self_w.detach().cpu())
            else:
                h_k, mem_candidate = self.rma(chunk, mem)

            # --- Cross-attention (Eq. 7–8) ---
            if return_attn:
                h_k, cross_w = self._cross_attention(
                    h_k, enc_chunk, enc_mem, return_attn=True
                )
                cross_attn_maps.append(cross_w.detach().cpu())
            else:
                h_k = self._cross_attention(h_k, enc_chunk, enc_mem)

            # --- Gated memory update *after* cross-attention (Section 3.1) ---
            mem = (self.gate(mem_candidate, mem) if start > 0
                   else mem_candidate)
            chunks_out.append(h_k)

        dec_out = torch.cat(chunks_out, dim=1) if chunks_out else x

        if return_attn:
            return dec_out, self_attn_maps, cross_attn_maps
        return dec_out


# ---------------------------------------------------------------------------
# RAILT  (Full Model — Section 3, Eq. 9)
# ---------------------------------------------------------------------------
class RAILT(nn.Module):
    """
    RAILT: Recurrent Memory AttentIon for Large-Scale Interaction
           Knowledge Tracing.

    Prediction:
        ŷ_t = σ(W_o · LN(h_t) + b_o)                   ... Eq. (9)
    """

    def __init__(self, d_model: int, num_enc_layers: int, num_dec_layers: int,
                 n_heads_enc: int, n_heads_dec: int,
                 total_ex: int, total_cat: int, total_res: int,
                 dropout: float, chunk_len: int = 50, mem_len: int = 16):
        super().__init__()

        self.encoders = nn.ModuleList([
            RAILTEncoder(d_model, n_heads_enc, total_ex, total_cat,
                         dropout, chunk_len, mem_len)
            for _ in range(num_enc_layers)
        ])
        self.decoders = nn.ModuleList([
            RAILTDecoder(d_model, total_res, n_heads_dec,
                         dropout, chunk_len, mem_len)
            for _ in range(num_dec_layers)
        ])

        # --- Eq. (9): output projection ---
        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, in_ex, in_cat, in_res, return_attn=False):
        """
        Args:
            in_ex:  problem IDs  (B, L)
            in_cat: skill IDs    (B, L)
            in_res: response IDs (B, L)
        Returns:
            pred: (B, L) — predicted probability of correct response
        """
        # === Encoder (Section 3.2) ===
        enc_out, enc_mem = None, None
        enc_attns = []

        for i, encoder in enumerate(self.encoders):
            first = (i == 0)
            if first and return_attn:
                enc_out, enc_mem, enc_attns = encoder(
                    in_ex, in_cat, first_block=True, return_attn=True
                )
            else:
                enc_out, enc_mem = encoder(
                    enc_out if not first else in_ex,
                    None if not first else in_cat,
                    first_block=first,
                )

        # === Decoder (Section 3.3) ===
        dec_out = None
        dec_self_attns, dec_cross_attns = [], []

        for i, decoder in enumerate(self.decoders):
            first = (i == 0)
            if first and return_attn:
                dec_out, dec_self_attns, dec_cross_attns = decoder(
                    in_res, enc_out, enc_mem,
                    first_block=True, return_attn=True,
                )
            else:
                dec_out = decoder(
                    dec_out if not first else in_res,
                    enc_out,
                    enc_mem if first else None,
                    first_block=first,
                )

        # === Eq. (9): ŷ_t = σ(W_o · LN(h_t) + b_o) ===
        pred = torch.sigmoid(self.out_proj(self.final_ln(dec_out))).squeeze(-1)

        if return_attn:
            return pred, enc_attns, dec_self_attns, dec_cross_attns
        return pred
