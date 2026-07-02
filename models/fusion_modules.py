import torch
import torch.nn as nn


class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-5, affine=True):
        super(LayerNorm1d, self).__init__()
        self.ln = nn.LayerNorm(channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class Fusion_Attention_Gate(nn.Module):
    def __init__(self, in_channels_ske, in_channels_rgb, out_channels, num_classes, num_heads=4):
        super(Fusion_Attention_Gate, self).__init__()
        self.ske_proj = nn.Conv1d(in_channels_ske, out_channels, kernel_size=1)
        self.rgb_proj = nn.Conv1d(in_channels_rgb, out_channels, kernel_size=1)

        assert out_channels % num_heads == 0
        self.attn_ske = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads)
        self.attn_rgb = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads)

        self.gate_ske = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            LayerNorm1d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate_rgb = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            LayerNorm1d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(0.1)
        self.post_fusion = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv1d(out_channels, num_classes, kernel_size=1)

    def _apply_gate(self, q, qa, gate_net):
        gate = gate_net(torch.cat([qa, q], dim=1))
        mod = gate * qa
        return mod + q

    def forward(self, ske, rgb, mask=None, return_features=False):
        feat_s = self.ske_proj(ske)
        feat_r = self.rgb_proj(rgb)

        q_s_t = feat_s.permute(2, 0, 1)
        q_r_t = feat_r.permute(2, 0, 1)

        key_padding_mask = None
        if mask is not None:
            mask_sq = mask[:, 0, :]
            key_padding_mask = (mask_sq == 0)

        s_attended, _ = self.attn_ske(
            query=q_s_t, key=q_r_t, value=q_r_t, key_padding_mask=key_padding_mask
        )
        r_attended, _ = self.attn_rgb(
            query=q_r_t, key=q_s_t, value=q_s_t, key_padding_mask=key_padding_mask
        )

        s_attended = self.dropout(s_attended).permute(1, 2, 0)
        r_attended = self.dropout(r_attended).permute(1, 2, 0)

        s_final = self._apply_gate(feat_s, s_attended, self.gate_ske)
        r_final = self._apply_gate(feat_r, r_attended, self.gate_rgb)

        fused = self.post_fusion(torch.cat([s_final, r_final], dim=1))
        out = self.conv_out(fused)

        if mask is not None:
            out = out * mask
        if return_features:
            return out, s_final, r_final
        return out
