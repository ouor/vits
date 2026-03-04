"""
텍스트 처리 패키지.

공개 API:
    - get_language(name)       → Language 인스턴스
    - list_languages()         → 등록된 언어 이름 목록
    - register_language(name)  → Language 구현체 등록 데코레이터
    - SymbolTable              → symbols ↔ ID 매핑
    - text_to_sequence(text, cleaner_names)   → list[int]  (하위 호환)
    - cleaned_text_to_sequence(cleaned_text)  → list[int]  (하위 호환)
    - sequence_to_text(sequence)              → str         (하위 호환)
"""
from __future__ import annotations

from vits.data.text.base import (  # noqa: F401
    Language,
    SymbolTable,
    get_language,
    list_languages,
    register_language,
)

# 모든 언어 구현을 자동 등록
from vits.data.text import languages as _languages  # noqa: F401

# ── 하위 호환 (backward compatibility) ───────────────────────────────────────
# 기존 text/__init__.py 와 동일한 함수 시그니처를 제공한다.
# 기존 코드가 `from vits.data.text import text_to_sequence` 형태로
# 사용할 수 있도록 유지한다.

import sys as _sys
import os as _os

# 기존 text 패키지 경로를 찾아 import
_ORIG_TEXT_DIR = _os.path.join(
    _os.path.dirname(__file__), "..", "..", "..", "..", ".."
)
_ORIG_TEXT_DIR = _os.path.normpath(_ORIG_TEXT_DIR)
if _ORIG_TEXT_DIR not in _sys.path:
    _sys.path.insert(0, _ORIG_TEXT_DIR)

try:
    from text import cleaners as _cleaners  # type: ignore[import]
    from text.symbols import symbols as _symbols  # type: ignore[import]

    _symbol_to_id: dict[str, int] = {s: i for i, s in enumerate(_symbols)}
    _id_to_symbol: dict[int, str] = {i: s for i, s in enumerate(_symbols)}

    def _clean_text(text: str, cleaner_names: list[str]) -> str:
        for name in cleaner_names:
            cleaner = getattr(_cleaners, name, None)
            if cleaner is None:
                raise ValueError(f"알 수 없는 cleaner: {name!r}")
            text = cleaner(text)
        return text

    def text_to_sequence(text: str, cleaner_names: list[str]) -> list[int]:
        """텍스트를 심볼 ID 시퀀스로 변환 (하위 호환)."""
        clean_text = _clean_text(text, cleaner_names)
        return [_symbol_to_id[s] for s in clean_text if s in _symbol_to_id]

    def cleaned_text_to_sequence(cleaned_text: str) -> list[int]:
        """정제된 텍스트를 심볼 ID 시퀀스로 변환 (하위 호환)."""
        return [_symbol_to_id[s] for s in cleaned_text if s in _symbol_to_id]

    def sequence_to_text(sequence: list[int]) -> str:
        """심볼 ID 시퀀스를 텍스트로 역변환 (하위 호환)."""
        return "".join(_id_to_symbol.get(sid, "") for sid in sequence)

except Exception:  # 원본 text 모듈 없어도 패키지 자체는 import 가능하게
    def text_to_sequence(text: str, cleaner_names: list[str]) -> list[int]:  # type: ignore[misc]
        raise NotImplementedError("원본 text 모듈을 찾을 수 없어 text_to_sequence를 사용할 수 없습니다.")

    def cleaned_text_to_sequence(cleaned_text: str) -> list[int]:  # type: ignore[misc]
        raise NotImplementedError("원본 text 모듈을 찾을 수 없어 cleaned_text_to_sequence를 사용할 수 없습니다.")

    def sequence_to_text(sequence: list[int]) -> str:  # type: ignore[misc]
        raise NotImplementedError("원본 text 모듈을 찾을 수 없어 sequence_to_text를 사용할 수 없습니다.")


__all__ = [
    "Language",
    "SymbolTable",
    "get_language",
    "list_languages",
    "register_language",
    "text_to_sequence",
    "cleaned_text_to_sequence",
    "sequence_to_text",
]
