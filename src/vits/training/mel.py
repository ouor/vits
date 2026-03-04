"""
Mel spectrogram 처리 유틸리티.

mel_basis와 hann_window를 모듈 수준 캐시로 관리하여
매번 재계산을 피한다.
"""
from __future__ import annotations

import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)

MAX_WAV_VALUE = 32768.0

# ── 모듈 수준 캐시 (device / dtype 별로 관리) ─────────────────────────────────
_mel_basis: dict[str, torch.Tensor] = {}
_hann_window: dict[str, torch.Tensor] = {}


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────
def dynamic_range_compression(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    """로그 다이나믹 레인지 압축."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    """로그 다이나믹 레인지 복원."""
    return torch.exp(x) / C


def spectral_normalize(magnitudes: torch.Tensor) -> torch.Tensor:
    return dynamic_range_compression(magnitudes)


def spectral_denormalize(magnitudes: torch.Tensor) -> torch.Tensor:
    return dynamic_range_decompression(magnitudes)


# ──────────────────────────────────────────────────────────────────────────────
# Spectrogram
# ──────────────────────────────────────────────────────────────────────────────
def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    """
    오디오 波形 → 선형 magnitude spectrogram.

    Args:
        y:            waveform [B, T] 또는 [1, T], 값 범위 [-1, 1]
        n_fft:        FFT 크기
        sampling_rate: 샘플링 레이트 (캐시 키에만 사용)
        hop_size:     hop length
        win_size:     window 크기
        center:       STFT center padding 여부

    Returns:
        spectrogram [B, n_fft//2+1, T_spec]
    """
    if torch.min(y) < -1.0:
        logger.debug("spectrogram_torch: min waveform value = %.4f", torch.min(y).item())
    if torch.max(y) > 1.0:
        logger.debug("spectrogram_torch: max waveform value = %.4f", torch.max(y).item())

    key = f"{win_size}_{y.dtype}_{y.device}"
    if key not in _hann_window:
        _hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    pad = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
    y = y.squeeze(1)

    spec = torch.stft(
        y, n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window[key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


# ──────────────────────────────────────────────────────────────────────────────
# Mel spectrogram
# ──────────────────────────────────────────────────────────────────────────────
def spec_to_mel(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    """
    선형 magnitude spectrogram → mel spectrogram.

    Args:
        spec: [B, n_fft//2+1, T]

    Returns:
        mel spectrogram [B, num_mels, T]
    """
    from librosa.filters import mel as librosa_mel_fn  # type: ignore[import]

    key = f"{fmax}_{spec.dtype}_{spec.device}"
    if key not in _mel_basis:
        mel_np = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        _mel_basis[key] = torch.from_numpy(mel_np).to(dtype=spec.dtype, device=spec.device)

    spec = torch.matmul(_mel_basis[key], spec)
    return spectral_normalize(spec)


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    center: bool = False,
) -> torch.Tensor:
    """
    오디오 波形 → mel spectrogram (one-step 편의 함수).

    Args:
        y:  waveform [B, T] 또는 [1, T], 값 범위 [-1, 1]

    Returns:
        mel spectrogram [B, num_mels, T_spec]
    """
    if torch.min(y) < -1.0:
        logger.debug("mel_spectrogram_torch: min waveform = %.4f", torch.min(y).item())
    if torch.max(y) > 1.0:
        logger.debug("mel_spectrogram_torch: max waveform = %.4f", torch.max(y).item())

    from librosa.filters import mel as librosa_mel_fn  # type: ignore[import]

    mel_key = f"{fmax}_{y.dtype}_{y.device}"
    win_key = f"{win_size}_{y.dtype}_{y.device}"

    if mel_key not in _mel_basis:
        mel_np = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        _mel_basis[mel_key] = torch.from_numpy(mel_np).to(dtype=y.dtype, device=y.device)
    if win_key not in _hann_window:
        _hann_window[win_key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    pad = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
    y = y.squeeze(1)

    spec = torch.stft(
        y, n_fft,
        hop_length=hop_size, win_length=win_size,
        window=_hann_window[win_key],
        center=center, pad_mode="reflect",
        normalized=False, onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(_mel_basis[mel_key], spec)
    return spectral_normalize(spec)
