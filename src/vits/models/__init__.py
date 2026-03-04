"""
모델 패키지.

기존 models.py 의 모든 클래스를 서브모듈로 분리하고
이 __init__.py에서 한꺼번에 re-export하여 하위 호환성을 유지한다.

사용 예시 (기존 코드와 동일):
    from vits.models import SynthesizerTrn, MultiPeriodDiscriminator
    # 또는 직접 서브모듈 임포트
    from vits.models.synthesizer import SynthesizerTrn
"""
from vits.models.predictor import (  # noqa: F401
    StochasticDurationPredictor,
    DurationPredictor,
)
from vits.models.encoder import (  # noqa: F401
    TextEncoder,
    PosteriorEncoder,
)
from vits.models.flow import ResidualCouplingBlock  # noqa: F401
from vits.models.generator import Generator  # noqa: F401
from vits.models.discriminator import (  # noqa: F401
    DiscriminatorP,
    DiscriminatorS,
    MultiPeriodDiscriminator,
)
from vits.models.synthesizer import SynthesizerTrn  # noqa: F401

__all__ = [
    "StochasticDurationPredictor",
    "DurationPredictor",
    "TextEncoder",
    "PosteriorEncoder",
    "ResidualCouplingBlock",
    "Generator",
    "DiscriminatorP",
    "DiscriminatorS",
    "MultiPeriodDiscriminator",
    "SynthesizerTrn",
]
