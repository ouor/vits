"""
영어 Language 구현.
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vits.data.text.base import Language, register_language


@register_language("english")
class English(Language):
    """
    영어 IPA 변환.
    기존 text/english.py의 english_to_ipa2()를 활용한다.
    """

    @property
    def name(self) -> str:
        return "english"

    @property
    def pad_symbol(self) -> str:
        return "_"

    @property
    def punctuation(self) -> str:
        return "!'(),.:;? "

    @property
    def letters(self) -> str:
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def text_to_phonemes(self, text: str) -> str:
        """
        영어 텍스트를 IPA로 변환.
        phonemizer 의존성 필요.
        """
        try:
            from text.english import english_to_ipa2  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "영어 처리를 위해 phonemizer가 필요합니다: "
                "pip install 'vits-tts[english]'"
            ) from e
        return english_to_ipa2(text)

    def get_cleaner_names(self) -> list[str]:
        return ["english_cleaners2"]
