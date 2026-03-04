"""
공통 pytest fixtures.
기존 코드(루트 패키지)를 그대로 임포트해서 동작을 검증한다.
리팩토링 과정에서 이 테스트가 깨지면 안 된다.
"""
import sys
import os
import json
import pytest
import torch
import numpy as np

# 루트 경로를 sys.path에 추가 (기존 코드 임포트)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─── 공통 설정 fixture ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_config_path():
    """샘플 config.json 경로."""
    path = os.path.join(ROOT, "trains", "sample", "config.json")
    if not os.path.exists(path):
        pytest.skip(f"샘플 config 없음: {path}")
    return path


@pytest.fixture(scope="session")
def sample_config(sample_config_path):
    """샘플 config.json을 HParams 객체로 로드."""
    sys.path.insert(0, ROOT)
    from utils import get_hparams_from_file
    return get_hparams_from_file(sample_config_path)


@pytest.fixture(scope="session")
def sample_symbols(sample_config):
    """config에서 심볼 리스트 읽기."""
    with open(os.path.join(ROOT, "trains", "sample", "config.json")) as f:
        raw = json.load(f)
    return raw.get("symbols", [])


# ─── 더미 배치 Tensor fixtures ────────────────────────────────────────────────

@pytest.fixture
def dummy_batch_single():
    """싱글 화자용 더미 배치 (batch=2)."""
    B, T_text, T_spec, T_audio = 2, 30, 150, 38400  # 22050Hz * ~1.7초
    return {
        "x": torch.randint(0, 66, (B, T_text)),
        "x_lengths": torch.LongTensor([T_text, 20]),
        "spec": torch.randn(B, 513, T_spec),
        "spec_lengths": torch.LongTensor([T_spec, 100]),
        "y": torch.randn(B, 1, T_audio),
        "y_lengths": torch.LongTensor([T_audio, 25600]),
    }


@pytest.fixture
def dummy_batch_multi(dummy_batch_single):
    """멀티 화자용 더미 배치 (speaker id 포함)."""
    batch = dict(dummy_batch_single)
    batch["sid"] = torch.LongTensor([0, 1])
    return batch


# ─── 더미 오디오 fixture ──────────────────────────────────────────────────────

@pytest.fixture
def dummy_audio_22k():
    """22050Hz 모노 더미 오디오 (2초)."""
    sr = 22050
    return np.random.randn(sr * 2).astype(np.float32), sr


@pytest.fixture
def dummy_audio_48k():
    """48000Hz 스테레오 더미 오디오 (리샘플/변환 테스트용)."""
    sr = 48000
    audio = np.random.randn(2, sr * 2).astype(np.float32)  # stereo
    return audio, sr


# ─── 기본 config dict fixture ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def minimal_config_dict():
    """최소한의 유효한 config 딕셔너리 (파일 불필요)."""
    return {
        "train": {
            "log_interval": 200,
            "eval_interval": 1000,
            "seed": 1234,
            "epochs": 10,
            "learning_rate": 2e-4,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": 4,
            "fp16_run": False,      # 테스트에선 FP32
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
        },
        "data": {
            "training_files": "trains/sample/filelist_train.txt.cleaned",
            "validation_files": "trains/sample/filelist_val.txt.cleaned",
            "text_cleaners": ["cjke_cleaners2"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "n_speakers": 0,
            "cleaned_text": True,
        },
        "model": {
            "inter_channels": 64,      # 테스트용 작은 크기
            "hidden_channels": 64,
            "filter_channels": 128,
            "n_heads": 2,
            "n_layers": 2,
            "kernel_size": 3,
            "p_dropout": 0.0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 128,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "gin_channels": 0,
        },
    }
