# ─── VITS TTS Project Makefile ───────────────────────────────────────────────
.PHONY: help install install-dev build-ext lint format type-check test test-fast clean

PYTHON  := python3
PIP     := uv pip
SRC     := src/vits
TESTS   := tests

help:  ## 사용 가능한 명령어 목록
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── 설치 ─────────────────────────────────────────────────────────────────────
install:  ## 기본 패키지 설치
	$(PIP) install -e .

install-dev:  ## 개발용 전체 설치 (테스트, 린터 포함)
	$(PIP) install -e ".[dev,cli,audio,all-langs]"

install-all:  ## 모든 의존성 설치
	$(PIP) install -e ".[all]"

# ─── 빌드 ─────────────────────────────────────────────────────────────────────
build-ext:  ## monotonic_align Cython 확장 빌드
	cd monotonic_align && mkdir -p monotonic_align && \
	$(PYTHON) setup.py build_ext --inplace && cd ..

# ─── 코드 품질 ────────────────────────────────────────────────────────────────
lint:  ## 린트 검사 (ruff)
	ruff check $(SRC) $(TESTS)

format:  ## 코드 자동 포맷 (ruff format)
	ruff format $(SRC) $(TESTS)
	ruff check --fix $(SRC) $(TESTS)

format-check:  ## 포맷 변경 사항만 확인 (CI용)
	ruff format --check $(SRC) $(TESTS)

type-check:  ## 타입 검사 (mypy)
	mypy $(SRC)

check: lint type-check  ## lint + type-check 한 번에

# ─── 테스트 ──────────────────────────────────────────────────────────────────
test:  ## 전체 테스트 실행
	pytest $(TESTS)

test-fast:  ## GPU/SlOW 테스트 제외하고 빠른 테스트만
	pytest $(TESTS) -m "not slow and not integration"

test-cov:  ## 커버리지 리포트 포함 테스트
	pytest $(TESTS) --cov=$(SRC) --cov-report=html
	@echo "커버리지 리포트: htmlcov/index.html"

test-one:  ## 특정 테스트 파일 실행 (예: make test-one FILE=tests/unit/test_config.py)
	pytest $(FILE) -v

# ─── 학습 단축키 ─────────────────────────────────────────────────────────────
train-sample:  ## 샘플 데이터로 학습 테스트
	$(PYTHON) -m vits.cli train \
		--config trains/sample/config.json \
		--model trains/sample/models

# ─── 정리 ─────────────────────────────────────────────────────────────────────
clean:  ## 빌드/캐시 파일 제거
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "정리 완료"
