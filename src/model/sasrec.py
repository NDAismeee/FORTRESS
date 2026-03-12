import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRec(nn.Module):
    def __init__(self, num_items, max_seq_len, hidden_units=64, num_heads=1, num_blocks=2, dropout_rate=0.5):
        super(SASRec, self).__init__()
        self.item_embedding = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_units)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_units)

        self.max_seq_len = max_seq_len
        self.hidden_units = hidden_units

        # === NEW: projection head for contrastive learning (2-layer MLP) ===
        self.proj = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.position_embedding.weight, std=0.01)
        # proj is fine with default init

    def forward(self, input_seq):
        # input_seq: [B, T]
        pos_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=input_seq.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(input_seq)

        seq_emb = self.item_embedding(input_seq) + self.position_embedding(pos_ids)
        seq_emb = self.dropout(seq_emb)
        attention_mask = input_seq != 0

        for block in self.blocks:
            seq_emb = block(seq_emb, attention_mask)

        seq_emb = self.layer_norm(seq_emb)
        return seq_emb  # [B, T, H]

    def predict(self, seq_output, item_emb):
        # seq_output: [B, T, H] or [B, H]
        # item_emb: [B, H] or [C, H]
        if seq_output.dim() == 3:
            lengths = (seq_output.abs().sum(dim=-1) != 0).sum(dim=1)
            idx = (lengths - 1).clamp(min=0)
            rows = torch.arange(seq_output.size(0), device=seq_output.device)
            seq_output = seq_output[rows, idx]  # [B, H]

        if item_emb.dim() == 2 and item_emb.size(0) == seq_output.size(0):
            return (seq_output * item_emb).sum(dim=-1)  # [B]
        elif item_emb.dim() == 2:
            return torch.matmul(seq_output, item_emb.t())  # [B, C]
        else:
            raise ValueError(f"Unexpected item_emb shape: {item_emb.shape}")

    # === NEW: projection utility for CL ===
    def project(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        x: [B, H] or [..., H]
        returns projected (and L2-normalized) embeddings with same leading dims.
        """
        z = self.proj(x)
        if normalize:
            z = F.normalize(z, dim=-1)
        return z


class TransformerBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(hidden_units)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, hidden_units),
            nn.Dropout(dropout_rate),
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_units)

    def forward(self, x, attention_mask):
        key_padding_mask = ~attention_mask
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.attn_layer_norm(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.ffn_layer_norm(x + ffn_output)
        return x
