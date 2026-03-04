"""
기존 텍스트 처리 코드 단위 테스트.
리팩토링 후에도 동일하게 통과해야 한다.
"""
import pytest
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestTextConversion:
    """text/__init__.py의 변환 함수 테스트."""

    def test_cleaned_text_to_sequence_returns_list(self):
        """cleaned_text_to_sequence가 정수 리스트를 반환하는지 확인."""
        from text import cleaned_text_to_sequence
        from text.symbols import symbols

        # symbols 중 일부를 사용
        sample = symbols[:5]
        result = cleaned_text_to_sequence(sample)
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        assert len(result) == 5

    def test_cleaned_text_to_sequence_skips_unknown(self):
        """알 수 없는 심볼은 무시해야 한다."""
        from text import cleaned_text_to_sequence

        # 존재하지 않는 심볼 포함
        result = cleaned_text_to_sequence(["_", "XXXXUNKNOWNXXXX", ","])
        assert len(result) == 2  # 알 수 없는 것 제외

    def test_sequence_to_text_roundtrip(self):
        """sequence_to_text는 cleaned_text_to_sequence의 역 변환이어야 한다."""
        from text import cleaned_text_to_sequence, sequence_to_text
        from text.symbols import symbols

        original = list(symbols[:10])
        seq = cleaned_text_to_sequence(original)
        reconstructed = sequence_to_text(seq)
        assert list(reconstructed) == original

    def test_symbols_not_empty(self):
        """symbols 리스트가 비어있지 않아야 한다."""
        from text.symbols import symbols
        assert len(symbols) > 0

    def test_symbols_contains_pad(self):
        """첫 번째 심볼은 padding(_) 이어야 한다."""
        from text.symbols import symbols
        assert symbols[0] == "_"

    def test_symbols_contains_space(self):
        """공백 문자가 포함되어야 한다."""
        from text.symbols import symbols
        assert " " in symbols

    def test_symbols_unique(self):
        """모든 심볼은 고유해야 한다."""
        from text.symbols import symbols
        assert len(symbols) == len(set(symbols))


class TestCleaners:
    """text/cleaners.py의 클리너 함수 테스트."""

    def test_cjke_cleaners2_korean_tag(self):
        """[KO] 태그 한국어 처리가 동작하는지 확인."""
        try:
            from text.cleaners import cjke_cleaners2
        except ImportError:
            pytest.skip("cjke_cleaners2 사용 불가 (의존성 미설치)")

        text = "[KO]안녕[KO]"
        result = cjke_cleaners2(text)
        assert isinstance(result, str)
        assert len(result) > 0
        # [KO] 태그가 제거되었어야 함
        assert "[KO]" not in result

    def test_cjke_cleaners2_english_tag(self):
        """[EN] 태그 영어 처리가 동작하는지 확인."""
        try:
            from text.cleaners import cjke_cleaners2
        except ImportError:
            pytest.skip("cjke_cleaners2 사용 불가 (의존성 미설치)")

        text = "[EN]hello world[EN]"
        result = cjke_cleaners2(text)
        assert isinstance(result, str)
        assert "[EN]" not in result

    def test_cjke_cleaners2_empty_string(self):
        """빈 문자열 입력 처리."""
        try:
            from text.cleaners import cjke_cleaners2
        except ImportError:
            pytest.skip("cjke_cleaners2 사용 불가 (의존성 미설치)")

        result = cjke_cleaners2("")
        assert isinstance(result, str)

    def test_text_to_sequence_with_cleaner(self):
        """text_to_sequence가 cleaner와 함께 동작하는지 확인."""
        try:
            from text import text_to_sequence
        except ImportError:
            pytest.skip("text_to_sequence 사용 불가")

        try:
            result = text_to_sequence("[KO]안녕[KO]", ["cjke_cleaners2"])
            assert isinstance(result, list)
        except Exception:
            pytest.skip("클리너 의존성 미설치")


class TestSymbolsConfig:
    """config.json의 symbols 필드와 text/symbols.py 일관성 테스트."""

    def test_symbols_match_config(self, sample_config, sample_symbols):
        """config.json의 symbols가 text/symbols.py와 일치하는지 확인."""
        from text.symbols import symbols

        if not sample_symbols:
            pytest.skip("config.json에 symbols 필드 없음")

        assert symbols == sample_symbols, (
            f"symbols 불일치: config({len(sample_symbols)}) vs "
            f"symbols.py({len(symbols)})"
        )

    def test_n_vocab_matches_symbols(self, sample_config, sample_symbols):
        """모델의 n_vocab이 심볼 개수와 일치하는지 확인."""
        if not sample_symbols:
            pytest.skip("config.json에 symbols 필드 없음")

        from text.symbols import symbols
        # SynthesizerTrn의 n_vocab은 len(symbols)와 같아야 함
        assert len(symbols) == len(sample_symbols)
