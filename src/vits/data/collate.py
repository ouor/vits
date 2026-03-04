"""
배치 콜레이트 함수 모음.

TextAudioDataset.__getitem__ 의 출력을 받아
패딩된 배치 텐서를 반환한다.
"""
from __future__ import annotations

import torch


# ──────────────────────────────────────────────────────────────────────────────
# 공통 유틸리티
# ──────────────────────────────────────────────────────────────────────────────
def _sort_by_spec_desc(batch, spec_idx: int = 1):
    """스펙트로그램 길이 내림차순으로 배치를 정렬한다."""
    _, ids_sorted = torch.sort(
        torch.LongTensor([item[spec_idx].size(1) for item in batch]),
        dim=0,
        descending=True,
    )
    return ids_sorted


# ──────────────────────────────────────────────────────────────────────────────
# 단일 화자 콜레이트
# ──────────────────────────────────────────────────────────────────────────────
class TextAudioCollate:
    """
    단일 화자 배치 콜레이트.

    입력 배치: list of (text, spec, wav)
    출력:
        text_padded    [B, T_text]
        text_lengths   [B]
        spec_padded    [B, n_mels, T_spec]
        spec_lengths   [B]
        wav_padded     [B, 1, T_wav]
        wav_lengths    [B]
        (ids_sorted    [B])  ← return_ids=True 일 때
    """

    def __init__(self, return_ids: bool = False) -> None:
        self.return_ids = return_ids

    def __call__(self, batch):
        ids_sorted = _sort_by_spec_desc(batch, spec_idx=1)
        B = len(batch)

        max_text_len = max(item[0].size(0) for item in batch)
        max_spec_len = max(item[1].size(1) for item in batch)
        max_wav_len  = max(item[2].size(1) for item in batch)
        n_mels       = batch[0][1].size(0)

        text_padded = torch.zeros(B, max_text_len, dtype=torch.long)
        spec_padded = torch.zeros(B, n_mels, max_spec_len)
        wav_padded  = torch.zeros(B, 1, max_wav_len)
        text_lengths = torch.zeros(B, dtype=torch.long)
        spec_lengths = torch.zeros(B, dtype=torch.long)
        wav_lengths  = torch.zeros(B, dtype=torch.long)

        for i, orig_idx in enumerate(ids_sorted):
            text, spec, wav = batch[orig_idx]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


# ──────────────────────────────────────────────────────────────────────────────
# 다중 화자 콜레이트
# ──────────────────────────────────────────────────────────────────────────────
class TextAudioSpeakerCollate:
    """
    다중 화자 배치 콜레이트.

    입력 배치: list of (text, spec, wav, sid)
    출력:
        text_padded    [B, T_text]
        text_lengths   [B]
        spec_padded    [B, n_mels, T_spec]
        spec_lengths   [B]
        wav_padded     [B, 1, T_wav]
        wav_lengths    [B]
        sid            [B]
        (ids_sorted    [B])  ← return_ids=True 일 때
    """

    def __init__(self, return_ids: bool = False) -> None:
        self.return_ids = return_ids

    def __call__(self, batch):
        ids_sorted = _sort_by_spec_desc(batch, spec_idx=1)
        B = len(batch)

        max_text_len = max(item[0].size(0) for item in batch)
        max_spec_len = max(item[1].size(1) for item in batch)
        max_wav_len  = max(item[2].size(1) for item in batch)
        n_mels       = batch[0][1].size(0)

        text_padded = torch.zeros(B, max_text_len, dtype=torch.long)
        spec_padded = torch.zeros(B, n_mels, max_spec_len)
        wav_padded  = torch.zeros(B, 1, max_wav_len)
        text_lengths = torch.zeros(B, dtype=torch.long)
        spec_lengths = torch.zeros(B, dtype=torch.long)
        wav_lengths  = torch.zeros(B, dtype=torch.long)
        sid          = torch.zeros(B, dtype=torch.long)

        for i, orig_idx in enumerate(ids_sorted):
            text, spec, wav, spk = batch[orig_idx]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            sid[i] = spk[0]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid
