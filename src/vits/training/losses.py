"""
VITS 학습 손실 함수.

모든 함수는 float32 안전 cast를 포함하여 half-precision 학습 환경에서도 동작한다.
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from torch.nn import functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Feature matching loss
# ──────────────────────────────────────────────────────────────────────────────
def feature_loss(
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    판별기 내부 feature map 간의 L1 거리.

    Args:
        fmap_r: 실제 오디오에 대한 판별기 feature map 리스트 (per-discriminator)
        fmap_g: 생성된 오디오에 대한 판별기 feature map 리스트

    Returns:
        스칼라 loss 텐서
    """
    loss: torch.Tensor = torch.zeros(1, device=fmap_r[0][0].device)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss * 2


# ──────────────────────────────────────────────────────────────────────────────
# Discriminator loss (Least-Squares GAN)
# ──────────────────────────────────────────────────────────────────────────────
def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[float], list[float]]:
    """
    판별기 Least-Squares GAN 손실.

    Returns:
        (total_loss, r_losses_per_disc, g_losses_per_disc)
    """
    loss: torch.Tensor = torch.zeros(1, device=disc_real_outputs[0].device)
    r_losses: list[float] = []
    g_losses: list[float] = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss = loss + r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


# ──────────────────────────────────────────────────────────────────────────────
# Generator loss (Least-Squares GAN)
# ──────────────────────────────────────────────────────────────────────────────
def generator_loss(
    disc_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    생성기 Least-Squares GAN 손실.

    Returns:
        (total_loss, gen_losses_per_disc)
    """
    loss: torch.Tensor = torch.zeros(1, device=disc_outputs[0].device)
    gen_losses: list[torch.Tensor] = []

    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss = loss + l

    return loss, gen_losses


# ──────────────────────────────────────────────────────────────────────────────
# KL divergence loss
# ──────────────────────────────────────────────────────────────────────────────
def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """
    사후 분포와 사전 분포 사이의 KL divergence (마스크 평균).

    Args:
        z_p:     flow 통과 후 잠재 변수   [B, H, T]
        logs_q:  사후 분포 log-scale     [B, H, T]
        m_p:     사전 분포 mean          [B, H, T]
        logs_p:  사전 분포 log-scale     [B, H, T]
        z_mask:  유효 time-step 마스크   [B, 1, T]

    Returns:
        마스크 평균 KL divergence (스칼라)
    """
    z_p    = z_p.float()
    logs_q = logs_q.float()
    m_p    = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask)
