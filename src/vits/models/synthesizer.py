"""
VITS 합성기: SynthesizerTrn (학습 + 추론 + 음성 변환).
"""
from __future__ import annotations

import math
import sys
import os

_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from torch import nn

import commons  # type: ignore[import]
import monotonic_align  # type: ignore[import]

from vits.models.encoder import TextEncoder, PosteriorEncoder
from vits.models.generator import Generator
from vits.models.flow import ResidualCouplingBlock
from vits.models.predictor import StochasticDurationPredictor, DurationPredictor


class SynthesizerTrn(nn.Module):
    """
    VITS 학습용 합성기.

    forward()  → 학습 시 사용 (spectrogram + text 입력)
    infer()    → 추론 시 사용 (text 입력만)
    voice_conversion() → 화자 변환
    """

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        n_speakers: int = 0,
        gin_channels: int = 0,
        use_sdp: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_vocab               = n_vocab
        self.spec_channels         = spec_channels
        self.inter_channels        = inter_channels
        self.hidden_channels       = hidden_channels
        self.filter_channels       = filter_channels
        self.n_heads               = n_heads
        self.n_layers              = n_layers
        self.kernel_size           = kernel_size
        self.p_dropout             = p_dropout
        self.resblock              = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates        = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes   = upsample_kernel_sizes
        self.segment_size          = segment_size
        self.n_speakers            = n_speakers
        self.gin_channels          = gin_channels
        self.use_sdp               = use_sdp

        # ── 서브모듈 ────────────────────────────────────────────────────────────
        self.enc_p = TextEncoder(
            n_vocab, inter_channels, hidden_channels,
            filter_channels, n_heads, n_layers, kernel_size, p_dropout,
        )
        self.dec = Generator(
            inter_channels, resblock, resblock_kernel_sizes,
            resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
            upsample_kernel_sizes, gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels,
            5, 1, 16, gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4,
            gin_channels=gin_channels,
        )

        if use_sdp:
            self.dp: nn.Module = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
        else:
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
            )

        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    # ── 학습 forward ──────────────────────────────────────────────────────────

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        """
        Args:
            x:         텍스트 token IDs     [B, T_text]
            x_lengths: 텍스트 길이           [B]
            y:         mel spectrogram       [B, n_mels, T_spec]
            y_lengths: spec 길이             [B]
            sid:       화자 ID (다중화자만)   [B]
        """
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 0 else None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent  = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw  = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        m_p    = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    # ── 추론 ──────────────────────────────────────────────────────────────────

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale: float = 1.0,
        length_scale: float = 1.0,
        noise_scale_w: float = 1.0,
        max_len: int | None = None,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 0 else None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)

        w       = torch.exp(logw) * x_mask * length_scale
        w_ceil  = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask    = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn      = commons.generate_path(w_ceil, attn_mask)

        m_p    = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z   = self.flow(z_p, y_mask, g=g, reverse=True)
        o   = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    # ── 음성 변환 ─────────────────────────────────────────────────────────────

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers > 0 이어야 합니다."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p    = self.flow(z, y_mask, g=g_src)
        z_hat  = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat  = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
