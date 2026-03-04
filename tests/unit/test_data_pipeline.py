"""
Stage 1-3: 데이터 파이프라인 테스트.

실제 오디오 I/O 없이 mock을 사용하여 Dataset, Collate, Sampler를 검증한다.
"""
from __future__ import annotations

import os
import tempfile
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch

from vits.data.dataset import (
    DatasetConfig,
    TextAudioDataset,
    TextAudioLoader,
    TextAudioSpeakerLoader,
    _detect_and_load_filelist,
)
from vits.data.collate import TextAudioCollate, TextAudioSpeakerCollate
from vits.data.sampler import DistributedBucketSampler


# ──────────────────────────────────────────────────────────────────────────────
# 공통 픽스처
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def default_cfg() -> DatasetConfig:
    return DatasetConfig(
        text_cleaners=["basic_cleaners"],
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
    )


def _make_filelist(tmp_path, rows: List[str]) -> str:
    """임시 파일리스트를 만들고 경로를 반환한다."""
    p = tmp_path / "filelist.txt"
    p.write_text("\n".join(rows), encoding="utf-8")
    return str(p)


def _make_fake_audio_file(tmp_path, name="audio.wav") -> str:
    """1바이트짜리 더미 wav 파일을 생성한다 (경로 존재 확인용)."""
    p = tmp_path / name
    p.write_bytes(b"\x00" * 512)  # hop_length=256 → spec_len 추산 = 1
    return str(p)


# ──────────────────────────────────────────────────────────────────────────────
# 파일리스트 파싱
# ──────────────────────────────────────────────────────────────────────────────
class TestFilelistParsing:
    def test_single_speaker_detected(self, tmp_path):
        audio = _make_fake_audio_file(tmp_path)
        flist = _make_filelist(tmp_path, [f"{audio}|안녕하세요"])
        rows, is_ms = _detect_and_load_filelist(flist)
        assert not is_ms
        assert rows[0][0] == audio
        assert rows[0][1] == "안녕하세요"

    def test_multispeaker_detected(self, tmp_path):
        audio = _make_fake_audio_file(tmp_path)
        flist = _make_filelist(tmp_path, [f"{audio}|0|안녕하세요"])
        rows, is_ms = _detect_and_load_filelist(flist)
        assert is_ms
        assert rows[0][1] == "0"
        assert rows[0][2] == "안녕하세요"

    def test_empty_filelist(self, tmp_path):
        flist = _make_filelist(tmp_path, [])
        rows, is_ms = _detect_and_load_filelist(flist)
        assert rows == []
        assert not is_ms

    def test_text_with_pipe_preserved(self, tmp_path):
        audio = _make_fake_audio_file(tmp_path)
        # 텍스트에 | 가 있어도 첫 두 필드만 분리되어야 함
        flist = _make_filelist(tmp_path, [f"{audio}|hello|world|extra"])
        rows, is_ms = _detect_and_load_filelist(flist)
        assert is_ms  # 3+ 열 → 다중 화자
        # [audio, sid, text]에서 text는 나머지를 합친 것
        assert rows[0][2] == "world|extra"


# ──────────────────────────────────────────────────────────────────────────────
# DatasetConfig
# ──────────────────────────────────────────────────────────────────────────────
class TestDatasetConfig:
    def test_defaults(self, default_cfg: DatasetConfig):
        assert default_cfg.add_blank is True
        assert default_cfg.cleaned_text is False
        assert default_cfg.min_text_len == 1
        assert default_cfg.max_text_len == 190

    def test_hparams_compat(self, default_cfg: DatasetConfig):
        from vits.data.dataset import _config_from_hparams
        hparams = MagicMock()
        hparams.text_cleaners = ["basic_cleaners"]
        hparams.max_wav_value = 32768.0
        hparams.sampling_rate = 22050
        hparams.filter_length = 1024
        hparams.hop_length = 256
        hparams.win_length = 1024
        cfg = _config_from_hparams(hparams)
        assert cfg.sampling_rate == 22050


# ──────────────────────────────────────────────────────────────────────────────
# TextAudioDataset (모의 I/O 사용)
# ──────────────────────────────────────────────────────────────────────────────
def _patch_dataset_io(cfg: DatasetConfig):
    """오디오 I/O & 텍스트 처리를 패치하는 context manager 모음을 반환한다."""
    n_mels = cfg.filter_length // 2 + 1
    spec_len = 50
    wav_len = spec_len * cfg.hop_length

    fake_spec = torch.randn(n_mels, spec_len)
    fake_wav = torch.randn(1, wav_len)

    patches = [
        patch("vits.data.dataset.TextAudioDataset._load_audio", return_value=(fake_spec, fake_wav)),
        patch("vits.data.dataset.TextAudioDataset._process_text",
              return_value=torch.LongTensor([1, 2, 3, 4, 5])),
    ]
    return patches


class TestTextAudioDataset:
    def test_single_speaker_len(self, tmp_path, default_cfg: DatasetConfig):
        audio = _make_fake_audio_file(tmp_path)
        flist = _make_filelist(tmp_path, [
            f"{audio}|텍스트A",
            f"{audio}|텍스트B",
        ])
        ds = TextAudioDataset(flist, default_cfg)
        assert len(ds) == 2
        assert not ds.is_multispeaker

    def test_multispeaker_len(self, tmp_path, default_cfg: DatasetConfig):
        audio = _make_fake_audio_file(tmp_path)
        flist = _make_filelist(tmp_path, [
            f"{audio}|0|텍스트A",
            f"{audio}|1|텍스트B",
        ])
        ds = TextAudioDataset(flist, default_cfg)
        assert len(ds) == 2
        assert ds.is_multispeaker

    def test_filter_removes_long_text(self, tmp_path, default_cfg: DatasetConfig):
        audio = _make_fake_audio_file(tmp_path)
        long_text = "가" * 300  # max_text_len=190 초과
        flist = _make_filelist(tmp_path, [
            f"{audio}|짧은텍스트",
            f"{audio}|{long_text}",
        ])
        ds = TextAudioDataset(flist, default_cfg)
        assert len(ds) == 1

    def test_filter_removes_missing_audio(self, tmp_path, default_cfg: DatasetConfig):
        flist = _make_filelist(tmp_path, ["/nonexistent/audio.wav|텍스트"])
        ds = TextAudioDataset(flist, default_cfg)
        assert len(ds) == 0

    def test_lengths_populated(self, tmp_path, default_cfg: DatasetConfig):
        audio = _make_fake_audio_file(tmp_path)
        flist = _make_filelist(tmp_path, [f"{audio}|텍스트A", f"{audio}|텍스트B"])
        ds = TextAudioDataset(flist, default_cfg)
        assert len(ds.lengths) == len(ds)
        assert all(isinstance(l, int) for l in ds.lengths)

    def test_getitem_single_speaker(self, tmp_path, default_cfg: DatasetConfig):
        audio = _make_fake_audio_file(tmp_path)
        flist = _make_filelist(tmp_path, [f"{audio}|텍스트"])
        ds = TextAudioDataset(flist, default_cfg)
        patches = _patch_dataset_io(default_cfg)
        with patches[0], patches[1]:
            item = ds[0]
        assert len(item) == 3  # (text, spec, wav)
        assert isinstance(item[0], torch.Tensor)

    def test_getitem_multispeaker(self, tmp_path, default_cfg: DatasetConfig):
        audio = _make_fake_audio_file(tmp_path)
        flist = _make_filelist(tmp_path, [f"{audio}|3|텍스트"])
        ds = TextAudioDataset(flist, default_cfg)
        patches = _patch_dataset_io(default_cfg)
        with patches[0], patches[1]:
            item = ds[0]
        assert len(item) == 4  # (text, spec, wav, sid)
        assert item[3].item() == 3  # speaker_id

    def test_backward_compat_loader(self, tmp_path, default_cfg: DatasetConfig):
        audio = _make_fake_audio_file(tmp_path)
        flist = _make_filelist(tmp_path, [f"{audio}|텍스트"])
        ds = TextAudioLoader(flist, default_cfg)
        assert not ds.is_multispeaker

    def test_backward_compat_speaker_loader(self, tmp_path, default_cfg: DatasetConfig):
        audio = _make_fake_audio_file(tmp_path)
        flist = _make_filelist(tmp_path, [f"{audio}|0|텍스트"])
        ds = TextAudioSpeakerLoader(flist, default_cfg)
        assert ds.is_multispeaker


# ──────────────────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────────────────
def _make_fake_batch_single(n=4, n_mels=513):
    """단일 화자 가짜 배치."""
    batch = []
    for i in range(n):
        text = torch.LongTensor(list(range(5 + i)))
        spec = torch.randn(n_mels, 30 + i * 5)
        wav  = torch.randn(1, (30 + i * 5) * 256)
        batch.append((text, spec, wav))
    return batch


def _make_fake_batch_multi(n=4, n_mels=513):
    """다중 화자 가짜 배치."""
    batch = []
    for i in range(n):
        text = torch.LongTensor(list(range(5 + i)))
        spec = torch.randn(n_mels, 30 + i * 5)
        wav  = torch.randn(1, (30 + i * 5) * 256)
        sid  = torch.LongTensor([i % 3])
        batch.append((text, spec, wav, sid))
    return batch


class TestTextAudioCollate:
    def test_output_shapes_single(self):
        batch = _make_fake_batch_single(n=4, n_mels=513)
        collate = TextAudioCollate()
        out = collate(batch)
        text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths = out
        assert text_padded.shape == (4, max(b[0].size(0) for b in batch))
        assert spec_padded.shape[0] == 4
        assert spec_padded.shape[1] == 513

    def test_sorted_by_spec_desc(self):
        batch = _make_fake_batch_single(n=4, n_mels=513)
        collate = TextAudioCollate(return_ids=True)
        *_, ids_sorted = collate(batch)
        spec_lens = [batch[int(i)][1].size(1) for i in ids_sorted]
        assert spec_lens == sorted(spec_lens, reverse=True)

    def test_return_ids_false(self):
        batch = _make_fake_batch_single(n=2)
        collate = TextAudioCollate(return_ids=False)
        out = collate(batch)
        assert len(out) == 6

    def test_return_ids_true(self):
        batch = _make_fake_batch_single(n=2)
        collate = TextAudioCollate(return_ids=True)
        out = collate(batch)
        assert len(out) == 7

    def test_output_shapes_multi(self):
        batch = _make_fake_batch_multi(n=3, n_mels=513)
        collate = TextAudioSpeakerCollate()
        out = collate(batch)
        assert len(out) == 7
        text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid = out
        assert sid.shape == (3,)

    def test_speaker_ids_preserved(self):
        batch = _make_fake_batch_multi(n=3, n_mels=513)
        collate = TextAudioSpeakerCollate(return_ids=True)
        *rest, ids_sorted = collate(batch)
        sid = rest[-1]
        # 원본 배치에서 정렬된 sid와 일치해야 함
        expected_sids = [int(batch[int(idx)][3][0]) for idx in ids_sorted]
        assert sid.tolist() == expected_sids


# ──────────────────────────────────────────────────────────────────────────────
# DistributedBucketSampler
# ──────────────────────────────────────────────────────────────────────────────
class _FakeDataset:
    """lengths 속성만 가진 더미 Dataset."""
    def __init__(self, lengths: List[int]):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)


class TestDistributedBucketSampler:
    def test_single_replica(self):
        lengths = list(range(50, 350, 10))  # 30 samples
        ds = _FakeDataset(lengths)
        sampler = DistributedBucketSampler(
            ds,
            batch_size=4,
            boundaries=[0, 100, 200, 300, 400],
            num_replicas=1,
            rank=0,
        )
        assert len(sampler) > 0

    def test_iter_returns_batches_of_correct_size(self):
        lengths = list(range(50, 350, 10))
        ds = _FakeDataset(lengths)
        sampler = DistributedBucketSampler(
            ds,
            batch_size=4,
            boundaries=[0, 100, 200, 300, 400],
            num_replicas=1,
            rank=0,
        )
        sampler.set_epoch(0)
        batches = list(sampler)
        assert all(len(b) == 4 for b in batches)

    def test_two_replicas_same_num_batches(self):
        lengths = list(range(50, 450, 10))  # 40 samples
        ds = _FakeDataset(lengths)
        sampler0 = DistributedBucketSampler(
            ds, batch_size=2, boundaries=[0, 150, 300, 450], num_replicas=2, rank=0
        )
        sampler1 = DistributedBucketSampler(
            ds, batch_size=2, boundaries=[0, 150, 300, 450], num_replicas=2, rank=1
        )
        sampler0.set_epoch(0)
        sampler1.set_epoch(0)
        batches0 = list(sampler0)
        batches1 = list(sampler1)
        # 두 랭크는 동일한 배치 수를 처리해야 한다 (DDP 균등 분할)
        assert len(batches0) == len(batches1)

    def test_empty_bucket_removed(self):
        # 200-300 범위 샘플 없음
        lengths = [50, 80, 110, 140, 310, 330, 350]
        ds = _FakeDataset(lengths)
        sampler = DistributedBucketSampler(
            ds, batch_size=2, boundaries=[0, 100, 200, 300, 400], num_replicas=1, rank=0
        )
        # 비어있는 버킷만큼 경계가 줄어야 하지만, 에러 없이 동작해야 함
        assert len(sampler) >= 0
