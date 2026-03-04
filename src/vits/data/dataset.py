"""
통합 TextAudio 데이터셋.

단일 화자 / 다중 화자 모드를 자동 감지한다.

파일리스트 형식:
    단일 화자: /path/to/audio.wav|text
    다중 화자: /path/to/audio.wav|speaker_id|text

사용 예시::

    from vits.data.dataset import TextAudioDataset

    # 단일 화자
    ds = TextAudioDataset("filelist.txt", hparams)

    # 다중 화자 (파일리스트가 spk id 컬럼을 포함하면 자동 감지)
    ds = TextAudioDataset("filelist_spk.txt", hparams)
"""
from __future__ import annotations

import os
import random
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.utils.data

logger = logging.getLogger(__name__)

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_ROOT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터셋 설정 (dataclass → HParams 교체 가능)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    """TextAudioDataset 생성에 필요한 최소 파라미터."""

    text_cleaners: list[str]
    max_wav_value: float
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    cleaned_text: bool = False
    add_blank: bool = True
    min_text_len: int = 1
    max_text_len: int = 190
    seed: int = 1234


def _config_from_hparams(hparams) -> DatasetConfig:
    """Legacy HParams 객체에서 DatasetConfig를 생성한다."""
    return DatasetConfig(
        text_cleaners=hparams.text_cleaners,
        max_wav_value=hparams.max_wav_value,
        sampling_rate=hparams.sampling_rate,
        filter_length=hparams.filter_length,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length,
        cleaned_text=getattr(hparams, "cleaned_text", False),
        add_blank=getattr(hparams, "add_blank", True),
        min_text_len=getattr(hparams, "min_text_len", 1),
        max_text_len=getattr(hparams, "max_text_len", 190),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 파일리스트 파싱
# ──────────────────────────────────────────────────────────────────────────────
def _detect_and_load_filelist(
    filelist_path: str,
) -> tuple[list[list[str]], bool]:
    """
    파일리스트를 읽고 단일/다중 화자 여부를 자동 감지한다.

    반환:
        (rows, is_multispeaker)
        rows: 단일화자 → [[audio_path, text], ...],
              다화자 → [[audio_path, speaker_id, text], ...]
        is_multispeaker: 다중 화자이면 True
    """
    rows: list[list[str]] = []
    with open(filelist_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            rows.append(parts)

    if not rows:
        return rows, False

    # 컬럼 수로 판단: 3열 = [audio, sid, text], 2열 = [audio, text]
    n_cols = len(rows[0])
    if n_cols >= 3:
        is_multispeaker = True
        # 정규화: 각 행을 [audio, sid, text] 3열로
        rows = [[r[0].strip(), r[1].strip(), "|".join(r[2:]).strip()] for r in rows]
    else:
        is_multispeaker = False
        rows = [[r[0].strip(), "|".join(r[1:]).strip()] for r in rows]

    return rows, is_multispeaker


# ──────────────────────────────────────────────────────────────────────────────
# 통합 데이터셋 클래스
# ──────────────────────────────────────────────────────────────────────────────
class TextAudioDataset(torch.utils.data.Dataset):
    """
    단일 화자 / 다중 화자 겸용 텍스트-오디오 데이터셋.

    파일리스트 형식을 자동으로 감지하여 적절한 모드로 동작한다.

    Args:
        filelist_path: 파일리스트 경로
        config: DatasetConfig 또는 HParams 호환 객체
        is_multispeaker: None이면 파일리스트 구조에서 자동 감지
    """

    def __init__(
        self,
        filelist_path: str,
        config,
        *,
        is_multispeaker: Optional[bool] = None,
    ) -> None:
        if isinstance(config, DatasetConfig):
            self.cfg = config
        else:
            self.cfg = _config_from_hparams(config)

        self._rows, detected_ms = _detect_and_load_filelist(filelist_path)
        self.is_multispeaker: bool = is_multispeaker if is_multispeaker is not None else detected_ms

        random.seed(self.cfg.seed)
        random.shuffle(self._rows)
        self._filter()

    # ── 필터링 ────────────────────────────────────────────────────────────────

    def _filter(self) -> None:
        """텍스트 길이 범위를 벗어나는 샘플을 제거하고, 버킷 샘플러용 spec 길이를 추산한다."""
        kept: list[list[str]] = []
        lengths: list[int] = []

        text_col = 2 if self.is_multispeaker else 1
        for row in self._rows:
            text = row[text_col]
            if not (self.cfg.min_text_len <= len(text) <= self.cfg.max_text_len):
                continue
            audiopath = row[0]
            if not os.path.exists(audiopath):
                logger.warning("오디오 파일 없음, 건너뜀: %s", audiopath)
                continue
            kept.append(row)
            # spec 길이 추산: 파일 크기 기반 (byte / 2채널 / hop_length)
            lengths.append(os.path.getsize(audiopath) // (2 * self.cfg.hop_length))

        logger.info(
            "데이터셋 필터링: %d → %d 샘플", len(self._rows), len(kept)
        )
        self._rows = kept
        self.lengths: list[int] = lengths

    # ── 오디오 처리 ───────────────────────────────────────────────────────────

    def _load_audio(self, audiopath: str) -> tuple[torch.Tensor, torch.Tensor]:
        """오디오를 로드하고 (spec, wav_norm) 쌍을 반환한다."""
        from utils import load_wav_to_torch  # type: ignore[import]
        from mel_processing import spectrogram_torch  # type: ignore[import]

        audio, sr = load_wav_to_torch(audiopath)
        if sr != self.cfg.sampling_rate:
            raise ValueError(
                f"샘플링 레이트 불일치: 파일={sr}Hz, 설정={self.cfg.sampling_rate}Hz "
                f"({audiopath})"
            )
        wav_norm = audio / self.cfg.max_wav_value
        wav_norm = wav_norm.unsqueeze(0)

        spec_path = audiopath.replace(".wav", ".spec.pt")
        if os.path.exists(spec_path):
            spec = torch.load(spec_path, weights_only=True)
        else:
            spec = spectrogram_torch(
                wav_norm,
                self.cfg.filter_length,
                self.cfg.sampling_rate,
                self.cfg.hop_length,
                self.cfg.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_path)

        return spec, wav_norm

    # ── 텍스트 처리 ───────────────────────────────────────────────────────────

    def _process_text(self, text: str) -> torch.Tensor:
        """텍스트를 심볼 ID tensor로 변환한다."""
        import commons  # type: ignore[import]

        if self.cfg.cleaned_text:
            from text import cleaned_text_to_sequence  # type: ignore[import]
            ids = cleaned_text_to_sequence(text)
        else:
            from text import text_to_sequence  # type: ignore[import]
            ids = text_to_sequence(text, self.cfg.text_cleaners)

        if self.cfg.add_blank:
            ids = commons.intersperse(ids, 0)

        return torch.LongTensor(ids)

    # ── Dataset 인터페이스 ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int):
        row = self._rows[index]
        if self.is_multispeaker:
            audiopath, sid_str, text = row[0], row[1], row[2]
            text_tensor = self._process_text(text)
            spec, wav = self._load_audio(audiopath)
            sid = torch.LongTensor([int(sid_str)])
            return text_tensor, spec, wav, sid
        else:
            audiopath, text = row[0], row[1]
            text_tensor = self._process_text(text)
            spec, wav = self._load_audio(audiopath)
            return text_tensor, spec, wav


# ──────────────────────────────────────────────────────────────────────────────
# 하위 호환 별칭
# ──────────────────────────────────────────────────────────────────────────────
class TextAudioLoader(TextAudioDataset):
    """TextAudioLoader → TextAudioDataset 하위 호환 별칭."""

    def __init__(self, audiopaths_and_text, hparams) -> None:
        super().__init__(audiopaths_and_text, hparams, is_multispeaker=False)


class TextAudioSpeakerLoader(TextAudioDataset):
    """TextAudioSpeakerLoader → TextAudioDataset 하위 호환 별칭."""

    def __init__(self, audiopaths_sid_text, hparams) -> None:
        super().__init__(audiopaths_sid_text, hparams, is_multispeaker=True)
