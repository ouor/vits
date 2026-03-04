"""
Language implementations.

이 모듈을 임포트하면 모든 @register_language 데코레이터가 실행되어
레지스트리에 자동 등록된다.
"""
from . import korean  # noqa: F401
from . import japanese  # noqa: F401
from . import english  # noqa: F401
from . import mandarin  # noqa: F401

__all__ = ["korean", "japanese", "english", "mandarin"]
