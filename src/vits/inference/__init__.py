"""
추론 패키지 — 학습된 VITS 모델로 TTS/VC를 수행한다.

공개 API:
    VITSSynthesizer: 텍스트 → 오디오 고수준 인터페이스
"""
from vits.inference.synthesizer import VITSSynthesizer  # noqa: F401

__all__ = ["VITSSynthesizer"]
