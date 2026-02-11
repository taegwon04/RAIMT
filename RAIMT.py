import torch
import torch.nn as nn


class RMA_Block(nn.Module):
    """
    RMA (Recurrent Memory Attention) Module
    
    [Memory | Current Chunk] → Pos Emb → Multi-Head Local Self-Attention → FFN → Split
    
    Returns:
        data_out: output tokens for current chunk
        mem_new: new memory candidate (before gating)
    """
    def __init__(self, dim_model, num_heads, dropout, chunk_len, mem_len):
        super().__init__()
        self.dim_model = dim_model
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        self.pos_emb = nn.Embedding(mem_len + chunk_len, dim_model)

        self.self_attn = nn.MultiheadAttention(
            dim_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * 4, dim_model),
        )
        self.ln_attn = nn.LayerNorm(dim_model)
        self.ln_ffn = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, chunk_data, mem_state, device, return_attn=False):
        B, Lc, D = chunk_data.shape

        # === Concatenate: [Memory | Current Chunk] ===
        combined = torch.cat([mem_state, chunk_data], dim=1)

        # === Position Embedding ===
        pos_ids = torch.arange(combined.size(1), device=device).unsqueeze(0)
        combined = combined + self.pos_emb(pos_ids)

        # === Causal Mask (data tokens only; memory can attend freely) ===
        total_len = combined.size(1)
        causal_mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
        causal_mask[self.mem_len:, self.mem_len:] = torch.triu(
            torch.ones(Lc, Lc, dtype=torch.bool, device=device), diagonal=1
        )

        # === Multi-Head Local Self-Attention ===
        normed = self.ln_attn(combined)
        attn_out, attn_weights = self.self_attn(
            normed, normed, normed,
            attn_mask=causal_mask,
            need_weights=return_attn,
            average_attn_weights=False
        )
        combined = combined + self.dropout(attn_out)

        # === Position-wise FFN Block ===
        normed = self.ln_ffn(combined)
        combined = combined + self.dropout(self.ffn(normed))

        # === Split: New Memory / Output Tokens ===
        mem_new = combined[:, :self.mem_len, :]
        data_out = combined[:, self.mem_len:, :]

        if return_attn:
            return data_out, mem_new, attn_weights
        return data_out, mem_new


class RAIMT_Encoder(nn.Module):
    """
    Encoder: ProblemID + Skill → RMA → Gated Memory Update
    """
    def __init__(self, dim_model, num_heads, total_ex, total_cat, seq_len,
                 dropout, chunk_len=50, mem_len=16):
        super().__init__()
        self.dim_model = dim_model
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        # === Input Embeddings ===
        self.embd_problem = nn.Embedding(total_ex, dim_model, padding_idx=0)
        self.embd_skill = nn.Embedding(total_cat, dim_model, padding_idx=0)

        # === Initial Memory Tokens ===
        self.mem_tokens = nn.Parameter(torch.zeros(1, mem_len, dim_model))
        nn.init.xavier_uniform_(self.mem_tokens)

        # === RMA Block ===
        self.rma = RMA_Block(dim_model, num_heads, dropout, chunk_len, mem_len)

        # === Gated Memory Update: g(·) ===
        self.gate_proj = nn.Linear(dim_model, dim_model)

    def _gated_memory_update(self, mem_new, mem_old):
        """
        g(·): Gated Memory Update
        next_mem = gate * mem_new + (1 - gate) * mem_old
        """
        gate = torch.sigmoid(self.gate_proj(mem_new))
        return gate * mem_new + (1 - gate) * mem_old

    def forward(self, in_ex, in_cat, first_block=True, return_attn=False):
        device = in_ex.device

        if first_block:
            out = self.embd_problem(in_ex) + self.embd_skill(in_cat)
            B, L = in_ex.shape
        else:
            out = in_ex
            B, L, _ = out.shape

        outputs, attn_maps = [], []
        mem_state = self.mem_tokens.expand(out.size(0), -1, -1)

        for start in range(0, L, self.chunk_len):
            end = min(start + self.chunk_len, L)
            chunk = out[:, start:end, :]
            if chunk.size(1) == 0:
                continue

            # === RMA Block ===
            if return_attn:
                data_out, mem_new, weights = self.rma(chunk, mem_state, device, return_attn=True)
                attn_maps.append(weights.detach().cpu())
            else:
                data_out, mem_new = self.rma(chunk, mem_state, device)

            # === Gated Memory Update ===
            if start > 0:  # first chunk: no previous memory to gate with
                mem_state = self._gated_memory_update(mem_new, mem_state)
            else:
                mem_state = mem_new

            outputs.append(data_out)

        final_out = torch.cat(outputs, dim=1) if outputs else out

        if return_attn:
            return final_out, mem_state, attn_maps
        return final_out, mem_state


class RAIMT_Decoder(nn.Module):
    """
    Decoder: Response → RMA → Cross-Attention → FFN → Gated Memory Update
    """
    def __init__(self, dim_model, total_res, num_heads, seq_len, dropout,
                 chunk_len=50, mem_len=16):
        super().__init__()
        self.dim_model = dim_model
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        # === Input Embedding ===
        self.embd_response = nn.Embedding(total_res, dim_model, padding_idx=0)

        # === Initial Memory Tokens ===
        self.mem_tokens = nn.Parameter(torch.zeros(1, mem_len, dim_model))
        nn.init.xavier_uniform_(self.mem_tokens)

        # === Memory Projection (encoder mem → decoder initial mem) ===
        self.mem_proj = nn.Linear(dim_model, dim_model)

        # === RMA Block (Self-Attention part) ===
        self.rma = RMA_Block(dim_model, num_heads, dropout, chunk_len, mem_len)

        # === Recurrent Memory Cross-Attention ===
        self.enc_mem_proj = nn.Linear(dim_model, dim_model)
        self.cross_attn = nn.MultiheadAttention(
            dim_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_cross_q = nn.LayerNorm(dim_model)
        self.ln_cross_kv = nn.LayerNorm(dim_model)

        # === Post Cross-Attention FFN ===
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * 4, dim_model),
        )
        self.ln_ffn = nn.LayerNorm(dim_model)

        # === Gated Memory Update: g(·) ===
        self.gate_proj = nn.Linear(dim_model, dim_model)

        self.dropout = nn.Dropout(dropout)

    def _gated_memory_update(self, mem_new, mem_old):
        """g(·): Gated Memory Update"""
        gate = torch.sigmoid(self.gate_proj(mem_new))
        return gate * mem_new + (1 - gate) * mem_old

    def _cross_attention(self, data_out, enc_chunk, enc_mem, device,
                         return_attn=False):
        """
        Recurrent Memory Cross-Attention
        Query: decoder data output
        Key/Value: [projected encoder memory | encoder chunk output]
        """
        Lc = data_out.size(1)
        query = self.ln_cross_q(data_out)

        if enc_mem is not None:
            proj_mem = self.enc_mem_proj(enc_mem)
            cross_kv = torch.cat([proj_mem, enc_chunk], dim=1)
            cross_kv = self.ln_cross_kv(cross_kv)

            M = proj_mem.size(1)
            cross_mask = torch.zeros(Lc, M + Lc, dtype=torch.bool, device=device)
            cross_mask[:, M:] = torch.triu(
                torch.ones(Lc, Lc, dtype=torch.bool, device=device), diagonal=1
            )
        else:
            cross_kv = self.ln_cross_kv(enc_chunk)
            cross_mask = torch.triu(
                torch.ones(Lc, Lc, dtype=torch.bool, device=device), diagonal=1
            )

        cross_out, cross_weights = self.cross_attn(
            query, cross_kv, cross_kv,
            attn_mask=cross_mask,
            need_weights=return_attn,
            average_attn_weights=False
        )
        data_out = data_out + self.dropout(cross_out)

        # Post cross-attention FFN
        normed = self.ln_ffn(data_out)
        data_out = data_out + self.dropout(self.ffn(normed))

        if return_attn:
            return data_out, cross_weights
        return data_out

    def forward(self, in_res, enc_out, enc_mem, first_block=True, return_attn=False):
        device = enc_out.device

        if first_block:
            out = self.embd_response(in_res)
            B, L = in_res.shape
        else:
            out = in_res
            B, L, _ = out.shape

        # === Initial Memory: project from encoder memory ===
        if enc_mem is not None:
            mem_state = self.mem_proj(enc_mem)
        else:
            mem_state = self.mem_tokens.expand(out.size(0), -1, -1)

        outputs, self_attn_maps, cross_attn_maps = [], [], []

        for start in range(0, L, self.chunk_len):
            end = min(start + self.chunk_len, L)
            chunk = out[:, start:end, :]
            enc_chunk = enc_out[:, start:end, :]
            if chunk.size(1) == 0:
                continue

            # === RMA Block (Self-Attention) ===
            if return_attn:
                data_out, mem_new, self_w = self.rma(chunk, mem_state, device, return_attn=True)
                self_attn_maps.append(self_w.detach().cpu())
            else:
                data_out, mem_new = self.rma(chunk, mem_state, device)

            # === Recurrent Memory Cross-Attention ===
            if return_attn:
                data_out, cross_w = self._cross_attention(
                    data_out, enc_chunk, enc_mem, device, return_attn=True
                )
                cross_attn_maps.append(cross_w.detach().cpu())
            else:
                data_out = self._cross_attention(
                    data_out, enc_chunk, enc_mem, device
                )

            # === Gated Memory Update ===
            if start > 0:
                mem_state = self._gated_memory_update(mem_new, mem_state)
            else:
                mem_state = mem_new

            outputs.append(data_out)

        final_out = torch.cat(outputs, dim=1) if outputs else out

        if return_attn:
            return final_out, self_attn_maps, cross_attn_maps
        return final_out


class RAIMT(nn.Module):
    def __init__(self, dim_model, num_en, num_de, heads_en, heads_de,
                 total_ex, total_cat, total_res, seq_len, dropout,
                 chunk_len=50, mem_len=16):
        super().__init__()
        self.encoders = nn.ModuleList([
            RAIMT_Encoder(dim_model, heads_en, total_ex, total_cat, seq_len,
                        dropout, chunk_len, mem_len)
            for _ in range(num_en)
        ])
        self.decoders = nn.ModuleList([
            RAIMT_Decoder(dim_model, total_res, heads_de, seq_len,
                        dropout, chunk_len, mem_len)
            for _ in range(num_de)
        ])
        self.final_ln = nn.LayerNorm(dim_model)
        self.out = nn.Linear(dim_model, 1)

    def forward(self, in_ex, in_cat, in_res, return_attn=False):
        # === Encoder ===
        enc_out, enc_mem = None, None
        enc_attns = []

        for i, encoder in enumerate(self.encoders):
            if i == 0 and return_attn:
                enc_out, enc_mem, enc_attns = encoder(
                    in_ex, in_cat, first_block=True, return_attn=True
                )
            else:
                enc_out, enc_mem = encoder(
                    enc_out if i > 0 else in_ex,
                    None if i > 0 else in_cat,
                    first_block=(i == 0)
                )

        # === Decoder ===
        dec_out = None
        dec_self_attns, dec_cross_attns = [], []

        for i, decoder in enumerate(self.decoders):
            if i == 0 and return_attn:
                dec_out, dec_self_attns, dec_cross_attns = decoder(
                    in_res, enc_out, enc_mem,
                    first_block=True, return_attn=True
                )
            else:
                dec_out = decoder(
                    dec_out if i > 0 else in_res,
                    enc_out,
                    enc_mem if i == 0 else None,
                    first_block=(i == 0)
                )

        # === Output: LN → Linear → Sigmoid ===
        pred = torch.sigmoid(self.out(self.final_ln(dec_out))).squeeze(-1)

        if return_attn:
            return pred, enc_attns, dec_self_attns, dec_cross_attns
        return pred
