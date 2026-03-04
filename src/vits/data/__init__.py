"""
데이터 처리 패키지.

공개 API:
    - TextAudioDataset       통합 데이터셋 (단일/다중 화자 자동 감지)
    - DatasetConfig          데이터셋 설정 dataclass
    - TextAudioCollate       단일 화자 콜레이트
    - TextAudioSpeakerCollate 다중 화자 콜레이트
    - DistributedBucketSampler 버킷 분산 샘플러
    - TextAudioLoader        하위 호환 별칭
    - TextAudioSpeakerLoader 하위 호환 별칭
"""
from vits.data.dataset import (  # noqa: F401
    TextAudioDataset,
    DatasetConfig,
    TextAudioLoader,
    TextAudioSpeakerLoader,
)
from vits.data.collate import (  # noqa: F401
    TextAudioCollate,
    TextAudioSpeakerCollate,
)
from vits.data.sampler import DistributedBucketSampler  # noqa: F401

__all__ = [
    "TextAudioDataset",
    "DatasetConfig",
    "TextAudioLoader",
    "TextAudioSpeakerLoader",
    "TextAudioCollate",
    "TextAudioSpeakerCollate",
    "DistributedBucketSampler",
]
