"""
일본어 Language 구현.
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


@register_language("japanese")
class Japanese(Language):
    """
    일본어 IPA 변환.
    기존 text/japanese.py의 japanese_to_ipa2()를 활용한다.
    """

    @property
    def name(self) -> str:
        return "japanese"

    @property
    def pad_symbol(self) -> str:
        return "_"

    @property
    def punctuation(self) -> str:
        return ",.!?-~…"

    @property
    def letters(self) -> str:
        return "AEINOQUabdefghijkmnoprstuvwyzʃʧʦ↓↑ "

    def text_to_phonemes(self, text: str) -> str:
        """
        일본어 텍스트를 IPA로 변환.
        pyopenjtalk 의존성 필요.
        """
        try:
            from text.japanese import japanese_to_ipa2  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "일본어 처리를 위해 pyopenjtalk가 필요합니다: "
                "pip install 'vits-tts[japanese]'"
            ) from e
        return japanese_to_ipa2(text)

    def get_cleaner_names(self) -> list[str]:
        return ["japanese_cleaners2"]
