"""
Stage 1-2: 텍스트 처리 플러그인 시스템 테스트.

Language ABC, register_language 데코레이터, SymbolTable, 레지스트리 API를 검증한다.
"""
from __future__ import annotations

import pytest
from vits.data.text.base import (
    Language,
    SymbolTable,
    get_language,
    list_languages,
    register_language,
)
from vits.data.text import languages as _lang_module  # noqa: F401 — 자동 등록 트리거


# ──────────────────────────────────────────────────────────────────────────────
# 픽스처: 테스트 전용 언어 등록
# ──────────────────────────────────────────────────────────────────────────────
@register_language("_test_lang")
class _TestLanguage(Language):
    @property
    def name(self) -> str:
        return "_test_lang"

    @property
    def pad_symbol(self) -> str:
        return "_"

    @property
    def punctuation(self) -> str:
        return ".,!"

    @property
    def letters(self) -> str:
        return "abcde"

    def text_to_phonemes(self, text: str) -> str:
        return text.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Language ABC
# ──────────────────────────────────────────────────────────────────────────────
class TestLanguageABC:
    def test_get_symbols_includes_pad_punctuation_letters(self):
        lang = _TestLanguage()
        syms = lang.get_symbols()
        assert "_" in syms
        assert "." in syms
        assert "a" in syms

    def test_get_symbols_no_duplicates(self):
        lang = _TestLanguage()
        syms = lang.get_symbols()
        assert len(syms) == len(set(syms)), "심볼 목록에 중복이 없어야 한다"

    def test_get_cleaner_names_returns_list(self):
        lang = _TestLanguage()
        cleaners = lang.get_cleaner_names()
        assert isinstance(cleaners, list)

    def test_text_to_phonemes_callable(self):
        lang = _TestLanguage()
        result = lang.text_to_phonemes("Hello")
        assert isinstance(result, str)

    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Language()  # type: ignore[abstract]


# ──────────────────────────────────────────────────────────────────────────────
# register_language + get_language
# ──────────────────────────────────────────────────────────────────────────────
class TestRegistry:
    def test_registered_language_retrievable(self):
        lang = get_language("_test_lang")
        assert isinstance(lang, _TestLanguage)

    def test_get_language_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="__nonexistent__"):
            get_language("__nonexistent__")

    def test_list_languages_contains_test_lang(self):
        langs = list_languages()
        assert "_test_lang" in langs

    def test_list_languages_returns_list_of_str(self):
        langs = list_languages()
        assert isinstance(langs, list)
        assert all(isinstance(n, str) for n in langs)

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="_test_lang"):
            @register_language("_test_lang")
            class _Dup(Language):
                @property
                def name(self):
                    return "_test_lang"

                @property
                def pad_symbol(self):
                    return "_"

                @property
                def punctuation(self):
                    return ""

                @property
                def letters(self):
                    return "x"

                def text_to_phonemes(self, text):
                    return text


# ──────────────────────────────────────────────────────────────────────────────
# 기본 내장 언어 등록 확인
# ──────────────────────────────────────────────────────────────────────────────
class TestBuiltinLanguages:
    @pytest.mark.parametrize("lang_name", ["korean", "japanese", "english", "mandarin"])
    def test_builtin_language_registered(self, lang_name: str):
        langs = list_languages()
        assert lang_name in langs, f"{lang_name}이 레지스트리에 등록되어 있어야 한다"

    @pytest.mark.parametrize("lang_name", ["korean", "japanese", "english", "mandarin"])
    def test_builtin_language_has_valid_symbols(self, lang_name: str):
        lang = get_language(lang_name)
        syms = lang.get_symbols()
        assert len(syms) > 5, f"{lang_name}의 심볼이 충분히 있어야 한다"
        assert lang.pad_symbol in syms

    @pytest.mark.parametrize("lang_name", ["korean", "japanese", "english", "mandarin"])
    def test_builtin_language_name_property(self, lang_name: str):
        lang = get_language(lang_name)
        assert lang.name == lang_name

    @pytest.mark.parametrize("lang_name", ["korean", "japanese", "english", "mandarin"])
    def test_builtin_language_cleaner_names_is_list(self, lang_name: str):
        lang = get_language(lang_name)
        assert isinstance(lang.get_cleaner_names(), list)


# ──────────────────────────────────────────────────────────────────────────────
# SymbolTable
# ──────────────────────────────────────────────────────────────────────────────
class TestSymbolTable:
    @pytest.fixture
    def table(self) -> SymbolTable:
        return SymbolTable(["_", "a", "b", "c", "!"])

    def test_len(self, table: SymbolTable):
        assert len(table) == 5

    def test_contains(self, table: SymbolTable):
        assert "a" in table
        assert "z" not in table

    def test_getitem_symbol_to_id(self, table: SymbolTable):
        idx = table["a"]
        assert isinstance(idx, int)
        assert 0 <= idx < len(table)

    def test_encode_decode_roundtrip(self, table: SymbolTable):
        text = "abc"
        ids = table.encode(text)
        assert ids == [table["a"], table["b"], table["c"]]
        decoded = table.decode(ids)
        assert decoded == text

    def test_encode_skips_unknown_symbols(self, table: SymbolTable):
        ids = table.encode("azb")  # 'z' 없음
        assert len(ids) == 2  # 'a', 'b'만

    def test_pad_id(self, table: SymbolTable):
        pad_id = table.pad_id
        assert pad_id == table["_"]

    def test_vocab_size(self, table: SymbolTable):
        assert table.vocab_size == 5

    def test_symbols_property(self, table: SymbolTable):
        syms = table.symbols
        assert isinstance(syms, list)
        assert len(syms) == 5

    def test_id_to_symbol_via_decode(self, table: SymbolTable):
        for sym in ["_", "a", "b", "c", "!"]:
            idx = table[sym]
            decoded = table.decode([idx])
            assert decoded == sym


# ──────────────────────────────────────────────────────────────────────────────
# vits.data.text 공개 API 하위 호환
# ──────────────────────────────────────────────────────────────────────────────
class TestTextPackagePublicAPI:
    def test_imports_available(self):
        from vits.data.text import (  # noqa: F401
            get_language,
            list_languages,
            register_language,
            SymbolTable,
            Language,
        )

    def test_backward_compat_functions_importable(self):
        from vits.data.text import (  # noqa: F401
            text_to_sequence,
            cleaned_text_to_sequence,
            sequence_to_text,
        )
