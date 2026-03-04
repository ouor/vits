"""
텍스트 처리 플러그인 시스템 기반 클래스.

Language 추상 클래스 + 레지스트리 패턴으로
새 언어를 플러그인 방식으로 추가할 수 있다.

사용 예시:
    # 언어 가져오기
    lang = get_language("korean")
    phonemes = lang.text_to_phonemes("[KO]안녕하세요[KO]")
    
    # 등록된 언어 목록
    print(list_languages())
    
    # 새 언어 등록
    @register_language("my_lang")
    class MyLanguage(Language):
        def symbols(self) -> list[str]: ...
        def text_to_phonemes(self, text: str) -> str: ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod

# ─── Language 추상 베이스 ───────────────────────────────────────────────────────

class Language(ABC):
    """
    언어별 텍스트 처리 플러그인의 추상 베이스 클래스.
    
    새 언어를 추가하려면 이 클래스를 상속하고
    @register_language("언어명") 데코레이터를 붙인다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """언어 식별자 (예: 'korean', 'japanese')."""
        ...

    @property
    @abstractmethod
    def pad_symbol(self) -> str:
        """패딩 심볼."""
        ...

    @property
    @abstractmethod
    def punctuation(self) -> str:
        """구두점 문자열 (각 문자가 심볼로 등록됨)."""
        ...

    @property
    @abstractmethod
    def letters(self) -> str:
        """음소/자모 문자열 (각 문자가 심볼로 등록됨)."""
        ...

    @abstractmethod
    def text_to_phonemes(self, text: str) -> str:
        """입력 텍스트를 음소 표현으로 변환한다."""
        ...

    # ─── 공통 구현 ─────────────────────────────────────────────────────────────

    def get_symbols(self) -> list[str]:
        """이 언어의 전체 심볼 리스트를 반환한다 (pad + 구두점 + 음소)."""
        return [self.pad_symbol] + list(self.punctuation) + list(self.letters)

    def get_cleaner_names(self) -> list[str]:
        """
        기존 cleaners.py 호환: cleaner 함수 이름 반환.
        하위 클래스에서 override해서 레거시 cleaner 이름을 반환할 수 있다.
        """
        return []

    def __repr__(self) -> str:
        return f"<Language: {self.name}, symbols={len(self.get_symbols())}>"


# ─── 레지스트리 ────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[Language]] = {}


def register_language(name: str):
    """
    Language 클래스를 전역 레지스트리에 등록하는 데코레이터.
    
    사용 예시:
        @register_language("korean")
        class Korean(Language):
            ...
    """
    def decorator(cls: type[Language]) -> type[Language]:
        if name in _REGISTRY:
            raise ValueError(
                f"언어 '{name}'이 이미 등록되어 있습니다: {_REGISTRY[name].__name__}"
            )
        if not issubclass(cls, Language):
            raise TypeError(f"{cls.__name__}은 Language를 상속해야 합니다.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_language(name: str) -> Language:
    """
    등록된 언어 인스턴스를 반환한다.
    
    raises:
        KeyError: 등록되지 않은 언어명인 경우
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"언어 '{name}'을 찾을 수 없습니다. "
            f"등록된 언어: {available}"
        )
    return _REGISTRY[name]()


def list_languages() -> list[str]:
    """등록된 모든 언어의 이름 목록을 반환한다."""
    return sorted(_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """해당 이름의 언어가 등록되어 있는지 확인한다."""
    return name in _REGISTRY


# ─── 심볼 관리 ─────────────────────────────────────────────────────────────────

class SymbolTable:
    """
    Language 인스턴스에서 심볼-ID 매핑을 관리한다.
    
    사용 예시:
        lang = get_language("korean")
        table = SymbolTable.from_language(lang)
        ids = table.encode("ㄱ ㄴ ㄷ")
    """

    def __init__(self, symbols: list[str], pad_symbol: str | None = None) -> None:
        if len(symbols) != len(set(symbols)):
            dupes = [s for s in set(symbols) if symbols.count(s) > 1]
            raise ValueError(f"심볼 중복 발견: {dupes}")
        self._symbols = list(symbols)
        self._pad_symbol: str = pad_symbol if pad_symbol is not None else (symbols[0] if symbols else "")
        self._sym2id: dict[str, int] = {s: i for i, s in enumerate(symbols)}
        self._id2sym: dict[int, str] = {i: s for i, s in enumerate(symbols)}

    @classmethod
    def from_language(cls, lang: Language) -> "SymbolTable":
        """Language 인스턴스의 심볼 리스트로 SymbolTable 생성."""
        return cls(lang.get_symbols(), pad_symbol=lang.pad_symbol)

    @classmethod
    def from_list(cls, symbols: list[str]) -> "SymbolTable":
        """심볼 리스트로 직접 생성."""
        return cls(symbols)

    # ── 속성 ───────────────────────────────────────────────────────────────────

    @property
    def symbols(self) -> list[str]:
        """전체 심볼 리스트."""
        return self._symbols

    @property
    def vocab_size(self) -> int:
        """어휘 크기 (심볼 개수)."""
        return len(self._symbols)

    @property
    def pad_id(self) -> int:
        """패딩 심볼의 ID."""
        return self._sym2id[self._pad_symbol]

    def encode(self, text: str) -> list[int]:
        """문자열 → 심볼 ID 리스트. 알 수 없는 심볼은 무시한다."""
        return [self._sym2id[ch] for ch in text if ch in self._sym2id]

    def decode(self, ids: list[int]) -> str:
        """심볼 ID 리스트 → 문자열."""
        return "".join(self._id2sym[i] for i in ids if i in self._id2sym)

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._sym2id

    def __getitem__(self, symbol: str) -> int:
        return self._sym2id[symbol]
