"""
Pydantic 기반 설정 스키마 단위 테스트.
"""
import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pydantic import ValidationError
from vits.configs.schema import (
    DataConfig,
    HParamsCompat,
    ModelConfig,
    TrainConfig,
    VITSConfig,
)


# ─── 최소 유효 설정 딕셔너리 ─────────────────────────────────────────────────

VALID_CONFIG = {
    "train": {
        "log_interval": 200,
        "eval_interval": 1000,
        "seed": 1234,
        "epochs": 100,
        "learning_rate": 2e-4,
        "betas": [0.8, 0.99],
        "eps": 1e-9,
        "batch_size": 8,
        "fp16_run": False,
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
        "inter_channels": 64,
        "hidden_channels": 64,
        "filter_channels": 128,
        "n_heads": 2,
        "n_layers": 2,
        "kernel_size": 3,
        "p_dropout": 0.0,
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [8, 8, 2, 2],      # prod = 256 = hop_length ✓
        "upsample_initial_channel": 128,
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "n_layers_q": 3,
        "use_spectral_norm": False,
        "gin_channels": 0,
    },
}


class TestTrainConfig:
    def test_valid_creation(self):
        tc = TrainConfig(**VALID_CONFIG["train"])
        assert tc.epochs == 100
        assert tc.batch_size == 8

    def test_default_values(self):
        """필수 필드가 없어도 기본값으로 생성."""
        tc = TrainConfig(
            training_files="a.txt",  # type: ignore
            validation_files="b.txt",
        )
        assert tc.epochs == 10000  # 기본값

    def test_invalid_learning_rate(self):
        with pytest.raises(ValidationError, match="learning_rate"):
            TrainConfig(**{**VALID_CONFIG["train"], "learning_rate": 5.0})

    def test_invalid_batch_size(self):
        with pytest.raises(ValidationError, match="batch_size"):
            TrainConfig(**{**VALID_CONFIG["train"], "batch_size": 0})

    def test_invalid_betas_length(self):
        with pytest.raises(ValidationError, match="betas"):
            TrainConfig(**{**VALID_CONFIG["train"], "betas": [0.8]})

    def test_invalid_betas_range(self):
        with pytest.raises(ValidationError, match="betas"):
            TrainConfig(**{**VALID_CONFIG["train"], "betas": [0.8, 1.5]})


class TestDataConfig:
    def test_valid_creation(self):
        dc = DataConfig(**VALID_CONFIG["data"])
        assert dc.sampling_rate == 22050
        assert dc.n_speakers == 0

    def test_invalid_sampling_rate(self):
        with pytest.raises(ValidationError, match="sampling_rate"):
            DataConfig(**{**VALID_CONFIG["data"], "sampling_rate": 8000})

    def test_invalid_n_speakers(self):
        with pytest.raises(ValidationError, match="n_speakers"):
            DataConfig(**{**VALID_CONFIG["data"], "n_speakers": -1})

    def test_mel_fmax_can_be_none(self):
        dc = DataConfig(**{**VALID_CONFIG["data"], "mel_fmax": None})
        assert dc.mel_fmax is None

    def test_mel_fmax_as_float(self):
        dc = DataConfig(**{**VALID_CONFIG["data"], "mel_fmax": 8000.0})
        assert dc.mel_fmax == 8000.0


class TestModelConfig:
    def test_valid_creation(self):
        mc = ModelConfig(**VALID_CONFIG["model"])
        assert mc.resblock == "1"

    def test_invalid_resblock(self):
        with pytest.raises(ValidationError, match="resblock"):
            ModelConfig(**{**VALID_CONFIG["model"], "resblock": "3"})

    def test_invalid_dropout(self):
        with pytest.raises(ValidationError, match="p_dropout"):
            ModelConfig(**{**VALID_CONFIG["model"], "p_dropout": 1.5})


class TestVITSConfig:
    def test_from_dict_valid(self):
        config = VITSConfig.from_dict(VALID_CONFIG)
        assert config.train.epochs == 100
        assert config.data.sampling_rate == 22050
        assert config.model.hidden_channels == 64

    def test_from_json_file(self, tmp_path):
        """JSON 파일에서 로드."""
        json_path = tmp_path / "config.json"
        json_path.write_text(json.dumps(VALID_CONFIG))

        config = VITSConfig.from_json(str(json_path))
        assert config.train.batch_size == 8

    def test_from_sample_config(self, sample_config_path):
        """실제 trains/sample/config.json 로드."""
        config = VITSConfig.from_json(sample_config_path)
        assert config.data.n_speakers == 2891
        assert config.data.sampling_rate == 22050
        assert config.model.gin_channels == 256

    def test_upsample_hop_consistency_fails(self):
        """upsample_rates 곱 != hop_length 이면 에러."""
        bad_config = dict(VALID_CONFIG)
        bad_config["model"] = dict(VALID_CONFIG["model"])
        bad_config["model"]["upsample_rates"] = [4, 4, 2, 2]  # prod=64 ≠ 256
        with pytest.raises(ValidationError, match="hop_length"):
            VITSConfig.from_dict(bad_config)

    def test_upsample_hop_consistency_passes(self):
        """upsample_rates 곱 == hop_length 이면 성공."""
        config = VITSConfig.from_dict(VALID_CONFIG)
        import math
        assert math.prod(config.model.upsample_rates) == config.data.hop_length

    def test_multi_speaker_requires_gin_channels(self):
        """n_speakers > 0인데 gin_channels == 0이면 에러."""
        bad_config = dict(VALID_CONFIG)
        bad_config["data"] = dict(VALID_CONFIG["data"], n_speakers=10)
        bad_config["model"] = dict(VALID_CONFIG["model"], gin_channels=0)
        with pytest.raises(ValidationError, match="gin_channels"):
            VITSConfig.from_dict(bad_config)

    def test_multi_speaker_with_gin_channels_passes(self):
        """n_speakers > 0 + gin_channels > 0 이면 성공."""
        good_config = dict(VALID_CONFIG)
        good_config["data"] = dict(VALID_CONFIG["data"], n_speakers=10)
        good_config["model"] = dict(VALID_CONFIG["model"], gin_channels=64)
        config = VITSConfig.from_dict(good_config)
        assert config.data.n_speakers == 10

    def test_extra_fields_allowed(self):
        """기존 config.json의 알 수 없는 필드는 무시 (하위 호환)."""
        extra_config = dict(VALID_CONFIG, unknown_field="some_value")
        # 에러 없이 생성되어야 함
        config = VITSConfig.from_dict(extra_config)
        assert config.train.epochs == 100

    def test_to_json_roundtrip(self, tmp_path):
        """저장 후 로드 시 동일한 config 반환."""
        config = VITSConfig.from_dict(VALID_CONFIG)
        json_path = tmp_path / "saved.json"
        config.to_json(str(json_path))

        reloaded = VITSConfig.from_json(str(json_path))
        assert reloaded.train.epochs == config.train.epochs
        assert reloaded.data.sampling_rate == config.data.sampling_rate
        assert reloaded.model.hidden_channels == config.model.hidden_channels

    def test_summary_contains_key_info(self):
        """summary()가 主요 정보를 포함하는지."""
        config = VITSConfig.from_dict(VALID_CONFIG)
        summary = config.summary()
        assert "22050" in summary       # sampling_rate
        assert "8" in summary           # batch_size
        assert "100" in summary         # epochs

    def test_symbols_optional(self):
        """symbols 필드 없이도 생성 가능."""
        config = VITSConfig.from_dict(VALID_CONFIG)
        assert config.symbols is None  # 기본값 None

    def test_symbols_from_config(self):
        """symbols 필드가 있으면 저장됨."""
        config_with_sym = dict(VALID_CONFIG, symbols=["_", "a", "b", "c"])
        config = VITSConfig.from_dict(config_with_sym)
        assert config.symbols == ["_", "a", "b", "c"]
        assert config.get_n_vocab() == 4


class TestHParamsCompat:
    """기존 HParams 호환 래퍼 테스트."""

    def test_from_vits_config(self):
        config = VITSConfig.from_dict(VALID_CONFIG)
        hp = config.to_hparams()
        assert isinstance(hp, HParamsCompat)

    def test_attribute_access(self):
        """속성으로 접근 가능."""
        config = VITSConfig.from_dict(VALID_CONFIG)
        hp = config.to_hparams()
        assert hp.train.epochs == 100
        assert hp.data.sampling_rate == 22050

    def test_contains_operator(self):
        """'in' 연산자 동작."""
        config = VITSConfig.from_dict(VALID_CONFIG)
        hp = config.to_hparams()
        assert "train" in hp
        assert "nonexistent" not in hp

    def test_items_iteration(self):
        """items() 동작."""
        config = VITSConfig.from_dict(VALID_CONFIG)
        hp = config.to_hparams()
        keys = [k for k, _ in hp.items()]
        assert "train" in keys
        assert "data" in keys
        assert "model" in keys
