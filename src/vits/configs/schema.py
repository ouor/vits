"""
VITS 설정 스키마.

Pydantic v2 기반으로 설정값의 타입 안전성과 유효성 검증을 제공한다.
기존 JSON config와 완전히 하위 호환된다.

사용 예시:
    # 기존 JSON에서 로드
    config = VITSConfig.from_json("trains/sample/config.json")
    
    # YAML에서 로드 (새 방식)
    config = VITSConfig.from_yaml("configs/experiment.yaml")
    
    # 직접 생성
    config = VITSConfig(train=TrainConfig(...), data=DataConfig(...), model=ModelConfig(...))
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


# ─── 학습 설정 ─────────────────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    log_interval: int = 200
    eval_interval: int = 1000
    seed: int = 1234
    epochs: int = 10000
    learning_rate: float = 2e-4
    betas: list[float] = [0.8, 0.99]
    eps: float = 1e-9
    batch_size: int = 32
    fp16_run: bool = True
    lr_decay: float = 0.999875
    segment_size: int = 8192
    init_lr_ratio: float = 1.0
    warmup_epochs: int = 0
    c_mel: float = 45.0
    c_kl: float = 1.0
    num_workers: int = 4
    log_dir: Optional[str] = None

    @field_validator("learning_rate")
    @classmethod
    def validate_lr(cls, v: float) -> float:
        if not (1e-6 <= v <= 1.0):
            raise ValueError(f"learning_rate {v}가 합리적인 범위(1e-6~1.0) 밖입니다.")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")
        return v

    @field_validator("betas")
    @classmethod
    def validate_betas(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("betas는 [beta1, beta2] 형태의 길이 2 리스트여야 합니다.")
        if not all(0.0 < b < 1.0 for b in v):
            raise ValueError("betas 값은 (0, 1) 사이여야 합니다.")
        return v


# ─── 데이터 설정 ───────────────────────────────────────────────────────────────

class DataConfig(BaseModel):
    training_files: str
    validation_files: str
    text_cleaners: list[str]
    max_wav_value: float = 32768.0
    sampling_rate: int = 22050
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None
    add_blank: bool = True
    n_speakers: int = 0       # 0 = 싱글 화자
    cleaned_text: bool = False
    min_text_len: int = 1
    max_text_len: int = 190

    @field_validator("sampling_rate")
    @classmethod
    def validate_sampling_rate(cls, v: int) -> int:
        supported = {16000, 22050, 24000, 44100, 48000}
        if v not in supported:
            raise ValueError(f"sampling_rate {v}Hz는 지원되지 않습니다. 지원: {supported}")
        return v

    @field_validator("n_speakers")
    @classmethod
    def validate_n_speakers(cls, v: int) -> int:
        if v < 0:
            raise ValueError("n_speakers는 0 이상이어야 합니다. (0 = 싱글 화자)")
        return v


# ─── 모델 설정 ─────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = [3, 7, 11]
    resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    upsample_rates: list[int] = [8, 8, 2, 2]
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = [16, 16, 4, 4]
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    gin_channels: int = 0
    use_sdp: bool = True

    @field_validator("resblock")
    @classmethod
    def validate_resblock(cls, v: str) -> str:
        if v not in ("1", "2"):
            raise ValueError(f"resblock은 '1' 또는 '2' 이어야 합니다, got: {v!r}")
        return v

    @field_validator("p_dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError(f"p_dropout은 [0, 1) 범위여야 합니다, got: {v}")
        return v


# ─── 전체 설정 ─────────────────────────────────────────────────────────────────

class VITSConfig(BaseModel):
    """VITS 전체 학습 설정."""

    train: TrainConfig
    data: DataConfig
    model: ModelConfig
    symbols: Optional[list[str]] = None  # config.json에서 직접 심볼 지정 시

    # 런타임 추가 필드 (직렬화 제외)
    model_dir: Optional[str] = None

    model_config = {"extra": "allow"}  # 알 수 없는 키를 허용 (하위 호환)

    @model_validator(mode="after")
    def validate_upsample_hop_consistency(self) -> "VITSConfig":
        """
        prod(upsample_rates) == hop_length 인지 검증.
        Generator의 업샘플링 배율이 hop_length와 일치해야 정확한 오디오가 생성된다.
        """
        upsample_product = math.prod(self.model.upsample_rates)
        hop_length = self.data.hop_length
        if upsample_product != hop_length:
            raise ValueError(
                f"upsample_rates의 곱({upsample_product})이 "
                f"hop_length({hop_length})와 일치하지 않습니다. "
                f"upsample_rates: {self.model.upsample_rates}"
            )
        return self

    @model_validator(mode="after")
    def validate_multi_speaker_gin_channels(self) -> "VITSConfig":
        """멀티 화자 설정 시 gin_channels가 0이 아니어야 함."""
        if self.data.n_speakers > 0 and self.model.gin_channels == 0:
            raise ValueError(
                f"n_speakers={self.data.n_speakers}이지만 model.gin_channels=0 입니다. "
                "멀티 화자 학습에는 gin_channels > 0이 필요합니다. (권장: 256)"
            )
        return self

    # ─── 팩토리 메서드 ──────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str | Path) -> "VITSConfig":
        """기존 JSON config 파일에서 로드 (하위 호환)."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls.model_validate(raw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VITSConfig":
        """YAML config 파일에서 로드."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "YAML 로드를 위해 pyyaml이 필요합니다: pip install pyyaml"
            ) from e

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, data: dict) -> "VITSConfig":
        """딕셔너리에서 직접 생성."""
        return cls.model_validate(data)

    # ─── 유틸리티 ──────────────────────────────────────────────────────────────

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """JSON 파일로 저장."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=indent))

    def to_hparams(self) -> "HParamsCompat":
        """기존 HParams 스타일 객체로 변환 (backward compatibility)."""
        return HParamsCompat.from_vits_config(self)

    def get_n_vocab(self) -> int:
        """n_vocab: 심볼 리스트 크기 (symbols 필드 또는 text.symbols에서)."""
        if self.symbols is not None:
            return len(self.symbols)
        # fallback: 현재 활성 symbols.py에서 읽기
        try:
            from text.symbols import symbols  # type: ignore[import]
            return len(symbols)
        except ImportError:
            raise RuntimeError(
                "n_vocab을 결정할 수 없습니다. "
                "config에 'symbols' 필드를 추가하거나 text/symbols.py를 설정하세요."
            )

    def estimate_memory_gb(self) -> float:
        """
        배치 크기와 모델 파라미터 기준으로 GPU 메모리 사용량을 대략 추정.
        실제 사용량과 다를 수 있으며 참고용입니다.
        """
        # 파라미터 수 추정 (Generator 기준 단순 추정)
        hidden = self.model.hidden_channels
        layers = self.model.n_layers
        param_estimate_m = (hidden * hidden * layers * 8) / 1e6  # 대략적인 파라미터 수 (M)

        # 배치 당 활성화 메모리 (seg_size * batch * channels * layers)
        batch = self.train.batch_size
        seg = self.train.segment_size
        activation_gb = (seg * batch * hidden * 4 * 4) / (1024**3)  # float32 기준

        # 파라미터 메모리 (FP16이면 2배 절약)
        fp16_factor = 0.5 if self.train.fp16_run else 1.0
        param_gb = (param_estimate_m * 1e6 * 4 * fp16_factor) / (1024**3)

        return round(param_gb + activation_gb, 1)

    def summary(self) -> str:
        """설정 요약 문자열 반환."""
        lines = [
            "─" * 50,
            "VITS Config Summary",
            "─" * 50,
            f"  배치 크기:      {self.train.batch_size}",
            f"  학습 에폭:      {self.train.epochs}",
            f"  학습률:         {self.train.learning_rate}",
            f"  FP16:           {self.train.fp16_run}",
            f"  샘플링 레이트:  {self.data.sampling_rate}Hz",
            f"  화자 수:        {self.data.n_speakers} ('0' = 싱글)",
            f"  Text cleaners:  {self.data.text_cleaners}",
            f"  Hidden channels:{self.model.hidden_channels}",
            f"  예상 GPU 메모리:~{self.estimate_memory_gb()}GB",
            "─" * 50,
        ]
        return "\n".join(lines)


# ─── 하위 호환 레이어 ───────────────────────────────────────────────────────────

class HParamsCompat:
    """
    기존 HParams 스타일 API를 제공하는 하위 호환 래퍼.
    새 코드에는 VITSConfig를 직접 사용할 것.
    """

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParamsCompat(**v)
            setattr(self, k, v)

    @classmethod
    def from_vits_config(cls, config: VITSConfig) -> "HParamsCompat":
        raw = json.loads(config.model_dump_json())
        hp = cls(**raw)
        if config.model_dir:
            hp.model_dir = config.model_dir
        return hp

    # HParams 인터페이스 호환
    def keys(self) -> list[str]:
        return list(self.__dict__.keys())

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def __setitem__(self, key: str, value: object) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __repr__(self) -> str:
        return repr(self.__dict__)
