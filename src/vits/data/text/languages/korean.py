"""
한국어 Language 구현.
"""
from __future__ import annotations

import sys
import os

# 루트 경로 추가 (기존 text/ 모듈 재사용)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vits.data.text.base import Language, register_language


@register_language("korean")
class Korean(Language):
    """
    한국어 IPA 변환.
    기존 text/korean.py의 korean_to_ipa()를 활용한다.
    """

    @property
    def name(self) -> str:
        return "korean"

    @property
    def pad_symbol(self) -> str:
        return "_"

    @property
    def punctuation(self) -> str:
        return ",.!?…~"

    @property
    def letters(self) -> str:
        # IPA 기반 cjke_cleaners2 심볼셋
        return "NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ "

    def text_to_phonemes(self, text: str) -> str:
        """
        한국어 텍스트를 IPA로 변환.
        [KO] 태그를 처리하거나 순수 Korean 텍스트를 변환한다.
        """
        try:
            from text.korean import korean_to_ipa  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "한국어 처리를 위해 ko-pron, jamo가 필요합니다: "
                "pip install 'vits-tts[korean]'"
            ) from e
        return korean_to_ipa(text)

    def get_cleaner_names(self) -> list[str]:
        return ["korean_cleaners"]
