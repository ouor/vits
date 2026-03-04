"""
VITSSynthesizer 단위 테스트.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from vits.inference.synthesizer import VITSSynthesizer
from vits.configs.schema import VITSConfig


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
def make_config(n_speakers: int = 0) -> VITSConfig:
    """최소한의 VITSConfig를 반환한다."""
    return VITSConfig.from_dict({
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "n_speakers": 0,  # model.n_speakers is unused; data.n_speakers is the real field
            "gin_channels": 256 if n_speakers > 0 else 0,
        },
        "data": {
            "training_files": "train.txt",
            "validation_files": "val.txt",
            "text_cleaners": ["english_cleaners2"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "n_speakers": n_speakers,
        },
        "train": {
            "log_interval": 200,
            "eval_interval": 1000,
            "seed": 1234,
            "epochs": 20000,
            "learning_rate": 2e-4,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": 4,
            "fp16_run": False,
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
        },
    })


def make_mock_model() -> MagicMock:
    """가짜 SynthesizerTrn을 반환한다."""
    model = MagicMock()
    # infer() 반환값: (audio, attn, mask, durations)
    B, T_wav = 1, 22050
    audio  = torch.zeros(B, 1, T_wav)
    attn   = torch.zeros(B, 1, 10, 10)
    mask   = torch.ones(B, 1, 10).bool()
    model.infer.return_value = (audio, attn, mask, None)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 팩토리 메서드 (_load_config)
# ──────────────────────────────────────────────────────────────────────────────
class TestLoadConfig:
    def test_accepts_vitconfig_directly(self):
        cfg = make_config()
        result = VITSSynthesizer._load_config(cfg, None)
        assert result is cfg

    def test_accepts_dict(self):
        cfg = make_config()
        d = {"model": cfg.model.model_dump(), "data": cfg.data.model_dump(), "train": cfg.train.model_dump()}
        result = VITSSynthesizer._load_config(d, None)
        assert isinstance(result, VITSConfig)

    def test_raises_if_no_config_and_no_file(self, tmp_path):
        ckpt = tmp_path / "G_100.pth"
        ckpt.touch()
        with pytest.raises(FileNotFoundError):
            VITSSynthesizer._load_config(None, ckpt)

    def test_finds_config_json_sibling(self, tmp_path):
        cfg = make_config()
        import json
        config_path = tmp_path / "config.json"
        # VITSConfig를 JSON으로 저장하는 최소 구조
        config_path.write_text(json.dumps({
            "model": cfg.model.model_dump(),
            "data": cfg.data.model_dump(),
            "train": cfg.train.model_dump(),
        }))
        ckpt = tmp_path / "G_100.pth"
        ckpt.touch()
        result = VITSSynthesizer._load_config(None, ckpt)
        assert isinstance(result, VITSConfig)


# ──────────────────────────────────────────────────────────────────────────────
# 합성 API
# ──────────────────────────────────────────────────────────────────────────────
class TestSynthetize:
    def _make_synthesizer(self, n_speakers: int = 0) -> VITSSynthesizer:
        cfg = make_config(n_speakers=n_speakers)
        model = make_mock_model()
        dev = torch.device("cpu")
        return VITSSynthesizer(model, cfg, dev)

    def test_synthesize_returns_numpy_array(self):
        syn = self._make_synthesizer()
        with patch.object(syn, "_preprocess_text", return_value=([1, 2, 3], "hello")):
            audio = syn.synthesize("hello")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32

    def test_synthesize_empty_text_returns_empty(self):
        syn = self._make_synthesizer()
        with patch.object(syn, "_preprocess_text", return_value=([], "")):
            audio = syn.synthesize("")
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 0

    def test_synthesize_batch_returns_list(self):
        syn = self._make_synthesizer()
        texts = ["hello", "world", "test"]
        with patch.object(syn, "_preprocess_text", return_value=([1, 2, 3], "x")):
            audios = syn.synthesize_batch(texts)
        assert isinstance(audios, list)
        assert len(audios) == len(texts)
        for a in audios:
            assert isinstance(a, np.ndarray)

    def test_synthesize_multispeaker_passes_sid(self):
        syn = self._make_synthesizer(n_speakers=4)
        with patch.object(syn, "_preprocess_text", return_value=([1, 2, 3], "hi")):
            syn.synthesize("hi", speaker_id=2)
        # model.infer가 sid를 받았는지 확인
        call_kwargs = syn.model.infer.call_args
        assert call_kwargs is not None
        # sid 텐서가 전달되어야 한다
        sid_arg = call_kwargs.kwargs.get("sid") or (
            call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
        )
        # sid 전달 여부 확인 (kwargs 혹은 positional)
        assert any(
            "sid" in str(call_kwargs)
            for call_kwargs in [syn.model.infer.call_args]
        )


# ──────────────────────────────────────────────────────────────────────────────
# 속성
# ──────────────────────────────────────────────────────────────────────────────
class TestProperties:
    def test_sampling_rate(self):
        cfg = make_config()
        syn = VITSSynthesizer(MagicMock(), cfg, torch.device("cpu"))
        assert syn.sampling_rate == 22050

    def test_is_multispeaker_false(self):
        cfg = make_config(n_speakers=0)
        syn = VITSSynthesizer(MagicMock(), cfg, torch.device("cpu"))
        assert not syn.is_multispeaker

    def test_is_multispeaker_true(self):
        cfg = make_config(n_speakers=4)
        syn = VITSSynthesizer(MagicMock(), cfg, torch.device("cpu"))
        assert syn.is_multispeaker
        assert syn.n_speakers == 4


# ──────────────────────────────────────────────────────────────────────────────
# _preprocess_text 기본 동작
# ──────────────────────────────────────────────────────────────────────────────
class TestPreprocessText:
    def test_returns_list_and_str(self):
        cfg = make_config()
        syn = VITSSynthesizer(MagicMock(), cfg, torch.device("cpu"))
        # text 모듈 없을 때 빈 시퀀스 반환
        with patch.dict("sys.modules", {"text": None}):
            ids, norm = syn._preprocess_text("hello")
        assert isinstance(ids, list)
        assert isinstance(norm, str)

    def test_with_text_module(self):
        """text 모듈이 있을 때 토큰 시퀀스를 반환한다."""
        cfg = make_config()
        syn = VITSSynthesizer(MagicMock(), cfg, torch.device("cpu"))

        mock_text = MagicMock()
        mock_text.text_to_sequence.return_value = [1, 0, 2, 0, 3]
        with patch.dict("sys.modules", {"text": mock_text}):
            ids, norm = syn._preprocess_text("hello")

        # add_blank=True (기본값) → intersperse 적용
        # commons.intersperse([1, 0, 2, 0, 3], 0)
        assert isinstance(ids, list)
        assert len(ids) > 0
