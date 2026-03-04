"""
VITS 추론 합성기 (Inference Synthesizer).

학습된 체크포인트를 로드해 텍스트에서 오디오를 생성하는 고수준 API.

기본 사용::

    from vits.inference import VITSSynthesizer

    syn = VITSSynthesizer.from_checkpoint("G_50000.pth", config="config.json")

    # 단일 화자
    audio = syn.synthesize("안녕하세요")

    # 다중 화자
    audio = syn.synthesize("Hello world", speaker_id=3)

    # 배치 합성
    audios = syn.synthesize_batch(["Hello", "World"], speaker_id=0)

    # numpy 배열 반환 (float32, shape [T])
    import soundfile as sf
    sf.write("out.wav", audio, syn.sampling_rate)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

# ── 루트 경로 설정 ────────────────────────────────────────────────────────────
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import commons  # type: ignore[import]

from vits.configs.schema import VITSConfig
from vits.models.synthesizer import SynthesizerTrn

logger = logging.getLogger(__name__)


class VITSSynthesizer:
    """
    VITS TTS 추론 인터페이스.

    Args:
        model:         학습된 SynthesizerTrn 모델 (eval 모드)
        config:        VITSConfig
        device:        실행 디바이스
    """

    def __init__(
        self,
        model: SynthesizerTrn,
        config: VITSConfig,
        device: torch.device,
    ) -> None:
        self.model   = model
        self.config  = config
        self.device  = device
        self.sampling_rate: int = config.data.sampling_rate

    # ── 팩토리 ────────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: Union[str, Path, VITSConfig, dict, None] = None,
        device: Optional[str] = None,
    ) -> "VITSSynthesizer":
        """
        체크포인트 파일에서 합성기를 생성한다.

        Args:
            checkpoint_path: G_*.pth 파일 경로
            config:          VITSConfig 객체, JSON 파일 경로, dict, 또는 None
                             None이면 checkpoint_path 옆 config.json을 시도한다.
            device:          'cuda', 'cpu', 'cuda:0' 등. None = 자동
        """
        ckpt_path = Path(checkpoint_path)
        cfg = cls._load_config(config, ckpt_path)

        if device is not None:
            dev = torch.device(device)
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        model = cls._build_model(cfg, dev)
        cls._load_weights(model, ckpt_path, dev)
        model.eval()

        logger.info("체크포인트 로드 완료: %s (device=%s)", ckpt_path.name, dev)
        return cls(model, cfg, dev)

    # ── 합성 API ──────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
        max_len: Optional[int] = None,
    ) -> np.ndarray:
        """
        텍스트를 오디오로 변환한다.

        Args:
            text:          입력 텍스트
            speaker_id:    화자 ID (다중 화자 모델에서 필수)
            noise_scale:   흐름 노이즈 강도 (높을수록 다양성 증가)
            noise_scale_w: 기간 예측기 노이즈 강도
            length_scale:  발화 속도 조절 (< 1 = 빠름, > 1 = 느림)
            max_len:       최대 출력 길이 (None = 무제한)

        Returns:
            float32 numpy 배열, shape [T], 값 범위 [-1, 1]
        """
        text_ids, text_norm = self._preprocess_text(text)
        return self._infer_single(
            text_ids, speaker_id,
            noise_scale, noise_scale_w, length_scale, max_len,
        )

    @torch.inference_mode()
    def synthesize_batch(
        self,
        texts: list[str],
        speaker_id: Optional[int] = None,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
        max_len: Optional[int] = None,
    ) -> list[np.ndarray]:
        """
        텍스트 목록을 일괄 처리해 오디오 목록을 반환한다.
        GPU OOM 방지를 위해 내부적으로 단건씩 처리한다.

        Returns:
            각 텍스트에 해당하는 float32 numpy 배열 목록
        """
        return [
            self.synthesize(
                t, speaker_id=speaker_id,
                noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                length_scale=length_scale, max_len=max_len,
            )
            for t in texts
        ]

    @torch.inference_mode()
    def voice_conversion(
        self,
        source_wav: np.ndarray,
        source_speaker_id: int,
        target_speaker_id: int,
    ) -> np.ndarray:
        """
        소스 오디오의 화자를 변환한다.

        Args:
            source_wav:        입력 오디오 (float32, shape [T])
            source_speaker_id: 소스 화자 ID
            target_speaker_id: 타겟 화자 ID

        Returns:
            변환된 float32 numpy 배열, shape [T]
        """
        from vits.training.mel import spectrogram_torch  # type: ignore

        cfg = self.config.data
        wav_tensor = torch.from_numpy(source_wav).unsqueeze(0).to(self.device)
        spec = spectrogram_torch(
            wav_tensor,
            cfg.filter_length,
            cfg.sampling_rate,
            cfg.hop_length,
            cfg.win_length,
            center=False,
        )
        spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)

        sid_src = torch.LongTensor([source_speaker_id]).to(self.device)
        sid_tgt = torch.LongTensor([target_speaker_id]).to(self.device)

        audio = self.model.voice_conversion(spec, spec_lengths, sid_src, sid_tgt)
        return audio[0, 0].cpu().numpy()

    # ── 속성 ──────────────────────────────────────────────────────────────────

    @property
    def is_multispeaker(self) -> bool:
        """다중 화자 모델이면 True."""
        return self.config.data.n_speakers > 0

    @property
    def n_speakers(self) -> int:
        """화자 수를 반환한다."""
        return self.config.data.n_speakers

    # ── 내부 구현 ─────────────────────────────────────────────────────────────

    def _preprocess_text(self, text: str) -> tuple[list[int], str]:
        """텍스트를 토큰 ID 목록으로 변환한다."""
        try:
            from text import text_to_sequence, cleaned_text_to_sequence  # type: ignore

            cleaners = self.config.data.text_cleaners
            use_cleaned = getattr(self.config.data, "cleaned_text", False)

            if use_cleaned:
                ids = cleaned_text_to_sequence(text)
                norm = text
            else:
                ids = text_to_sequence(text, cleaners)
                norm = text

            if getattr(self.config.model, "add_blank", True):
                ids = commons.intersperse(ids, 0)
            return ids, norm
        except ImportError:
            logger.warning("text 모듈을 찾을 수 없습니다. 빈 시퀀스를 반환합니다.")
            return [], text

    def _infer_single(
        self,
        text_ids: list[int],
        speaker_id: Optional[int],
        noise_scale: float,
        noise_scale_w: float,
        length_scale: float,
        max_len: Optional[int],
    ) -> np.ndarray:
        if not text_ids:
            return np.zeros(0, dtype=np.float32)

        x = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(text_ids)]).to(self.device)
        sid = (
            torch.LongTensor([speaker_id]).to(self.device)
            if speaker_id is not None
            else None
        )

        audio, attn, mask, _ = self.model.infer(
            x, x_lengths,
            sid=sid,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            max_len=max_len,
        )
        return audio[0, 0].cpu().float().numpy()

    @staticmethod
    def _load_config(
        config: Union[str, Path, VITSConfig, dict, None],
        ckpt_path: Path,
    ) -> VITSConfig:
        if isinstance(config, VITSConfig):
            return config
        if isinstance(config, dict):
            return VITSConfig.from_dict(config)
        if config is None:
            # 체크포인트 옆 config.json 탐색
            for candidate in [
                ckpt_path.parent / "config.json",
                ckpt_path.parent.parent / "config.json",
            ]:
                if candidate.exists():
                    return VITSConfig.from_json(str(candidate))
            raise FileNotFoundError(
                f"config.json을 찾을 수 없습니다. "
                f"config= 매개변수로 직접 지정하세요."
            )
        # 파일 경로
        return VITSConfig.from_json(str(config))

    @staticmethod
    def _build_model(cfg: VITSConfig, device: torch.device) -> SynthesizerTrn:
        n_vocab     = cfg.get_n_vocab()
        spec_ch     = cfg.data.filter_length // 2 + 1
        seg_size    = cfg.train.segment_size // cfg.data.hop_length
        model_kw    = cfg.model.model_dump()
        model = SynthesizerTrn(n_vocab, spec_ch, seg_size, **model_kw).to(device)
        return model

    @staticmethod
    def _load_weights(
        model: SynthesizerTrn,
        ckpt_path: Path,
        device: torch.device,
    ) -> None:
        """체크포인트에서 생성기 가중치만 로드한다."""
        try:
            import utils  # type: ignore[import]
            utils.load_checkpoint(str(ckpt_path), model, None)
        except Exception:
            # utils 없을 때 직접 로드
            state = torch.load(str(ckpt_path), map_location=device)
            if "model" in state:
                model.load_state_dict(state["model"])
            else:
                model.load_state_dict(state)
