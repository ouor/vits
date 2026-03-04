"""
중국어(표준 중국어, 만다린) Language 구현.
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


@register_language("mandarin")
class Mandarin(Language):
    """
    표준 중국어(보통화) IPA 변환.
    기존 text/mandarin.py의 chinese_to_ipa()를 활용한다.
    """

    @property
    def name(self) -> str:
        return "mandarin"

    @property
    def pad_symbol(self) -> str:
        return "_"

    @property
    def punctuation(self) -> str:
        return ",.!?…~，。！？…"

    @property
    def letters(self) -> str:
        return (
            "abcdefghijklmnopqrstuvwxyz"
            "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃ"
            " ˈˌ↓↑"
        )

    def text_to_phonemes(self, text: str) -> str:
        """
        중국어 텍스트를 IPA로 변환.
        pypinyin 의존성 필요.
        """
        try:
            from text.mandarin import chinese_to_ipa  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "중국어 처리를 위해 pypinyin이 필요합니다: "
                "pip install 'vits-tts[chinese]'"
            ) from e
        return chinese_to_ipa(text)

    def get_cleaner_names(self) -> list[str]:
        return ["chinese_cleaners"]
