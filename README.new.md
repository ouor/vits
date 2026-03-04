# VITS TTS — 리팩토링 프레임워크

원본 연구 코드(`train.py`, `train_ms.py`)를 프로덕션 수준의 패키지로 재구성한 결과물이다.  
기존 JSON config와 **완전히 하위 호환**되며, 기존 체크포인트(.pth)를 그대로 사용할 수 있다.

---

## 설치

```bash
# 기본 설치 (학습 + 추론)
pip install -e .

# 언어별 의존성 추가
pip install -e ".[korean]"          # 한국어
pip install -e ".[japanese]"        # 일본어
pip install -e ".[english]"         # 영어
pip install -e ".[all-langs]"       # 전체 언어

# 개발 환경
pip install -e ".[dev]"
```

### monotonic_align 빌드 (최초 1회)

```bash
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
cd ..
```

---

## 빠른 시작

### 1. 설정 파일 생성

```bash
# 단일 화자
vits init --output config.json

# 다중 화자 (예: 4명)
vits init --output config.json --speakers 4
```

생성된 `config.json`에서 다음 항목을 수정한다:

```json
{
  "model_dir": "./logs/my_model",
  "data": {
    "training_files": "filelists/train.txt.cleaned",
    "validation_files": "filelists/val.txt.cleaned",
    "sampling_rate": 22050
  },
  "train": {
    "batch_size": 16,
    "epochs": 20000
  }
}
```

### 2. 데이터 준비

**단일 화자** (`n_speakers = 0`):
```
path/to/audio.wav|transcript
trains/sample/001.wav|안녕하세요.
```

**다중 화자** (`n_speakers > 0`, speaker id는 0부터 시작):
```
path/to/audio.wav|speaker_id|transcript
trains/sample/001.wav|0|안녕하세요.
trains/sample/002.wav|1|Hello world.
```

### 3. 텍스트 전처리

```bash
vits prepare filelists/train.txt --config config.json
vits prepare filelists/val.txt   --config config.json
# → filelists/train.txt.cleaned, filelists/val.txt.cleaned 생성
```

또는 원본 스크립트로:

```bash
# 단일 화자 (text column index = 1)
python preprocess.py --text_index 1 --filelists filelists/train.txt filelists/val.txt

# 다중 화자 (text column index = 2)
python preprocess.py --text_index 2 --filelists filelists/train.txt filelists/val.txt
```

### 4. 학습

**CLI (권장):**

```bash
vits train --config config.json
```

**Python API:**

```python
from vits.configs import VITSConfig
from vits.training import VITSTrainer

config = VITSConfig.from_json("config.json")
trainer = VITSTrainer(config)
trainer.train()
```

**콜백 사용:**

```python
from vits.training import VITSTrainer, ModelCheckpoint, EarlyStopping, TQDMCallback

config = VITSConfig.from_json("config.json")
trainer = VITSTrainer(config, callbacks=[
    TQDMCallback(),
    ModelCheckpoint(save_dir="checkpoints", save_every_n_epochs=5),
    EarlyStopping(monitor="loss/g/total", patience=10),
])
trainer.train()
```

**분산 학습 (torchrun):**

```bash
torchrun --nproc_per_node=4 -m vits.training.launch --config config.json
```

또는 직접:

```python
import torch.distributed as dist
from vits.training import VITSTrainer

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
trainer = VITSTrainer(config, rank=rank, world_size=dist.get_world_size())
trainer.train()
dist.destroy_process_group()
```

**사전 학습 모델 재개:**  
`config.json`의 `model_dir`에 `G_*.pth`, `D_*.pth`를 넣으면 자동으로 재개한다.

---

## 추론

### CLI

```bash
# 단일 화자
vits infer --checkpoint logs/my_model/G_50000.pth --text "안녕하세요" --output out.wav

# 다중 화자 (화자 ID 지정)
vits infer --checkpoint logs/my_model/G_50000.pth --text "Hello" --speaker 2 --output out.wav

# 속도/음질 조절
vits infer --checkpoint G_50000.pth --text "Hello" \
    --noise-scale 0.667 \
    --noise-scale-w 0.8 \
    --length-scale 1.0
```

### Python API

```python
from vits.inference import VITSSynthesizer

# 로드 (config는 checkpoint 옆 config.json 자동 탐색)
syn = VITSSynthesizer.from_checkpoint("logs/my_model/G_50000.pth")

# 단건 합성
audio = syn.synthesize("안녕하세요")          # np.ndarray, float32

# 다중 화자
audio = syn.synthesize("Hello", speaker_id=2)

# 배치 합성
audios = syn.synthesize_batch(["Hello", "World", "Test"], speaker_id=0)

# 파일 저장
import soundfile as sf
sf.write("out.wav", audio, syn.sampling_rate)

# 화자 변환 (Voice Conversion)
import numpy as np
source_wav, _ = sf.read("source.wav", dtype="float32")
converted = syn.voice_conversion(source_wav, source_speaker_id=0, target_speaker_id=1)
sf.write("converted.wav", converted, syn.sampling_rate)
```

---

## 환경 진단

```bash
vits doctor
```

```
============================================================
VITS 환경 진단
============================================================
[시스템]
  Python:   3.11.14
  OS:       Linux 5.15.0

[패키지]
  ✓ PyTorch      2.8.0+cu126
  ✓ NumPy        2.3.5
  ...

[CUDA]
  ✓ CUDA 사용 가능: 12.6
  GPU 수: 1
    [0] NVIDIA GeForce RTX 3090 (25.3 GB)
```

---

## TensorBoard

`config.json`의 `train.log_dir`를 설정하면 자동으로 기록된다.

```bash
tensorboard --logdir=logs/my_model/tb
```

---

## 패키지 구조

```
src/vits/
├── configs/
│   └── schema.py          # VITSConfig, ModelConfig, DataConfig, TrainConfig (Pydantic v2)
├── data/
│   ├── dataset.py         # TextAudioDataset (단일/다중 화자 통합)
│   ├── collate.py         # TextAudioCollate, TextAudioSpeakerCollate
│   ├── sampler.py         # DistributedBucketSampler
│   └── text/
│       ├── base.py        # Language ABC, SymbolTable
│       └── languages/     # @register_language 플러그인
│           ├── korean.py
│           ├── japanese.py
│           ├── english.py
│           └── mandarin.py
├── models/
│   ├── synthesizer.py     # SynthesizerTrn (forward / infer / voice_conversion)
│   ├── encoder.py         # TextEncoder, PosteriorEncoder
│   ├── decoder.py         # Generator (HiFi-GAN)
│   ├── flow.py            # ResidualCouplingBlock
│   ├── predictor.py       # StochasticDurationPredictor, DurationPredictor
│   └── discriminator.py   # MultiPeriodDiscriminator
├── training/
│   ├── trainer.py         # VITSTrainer (단일/다중 GPU, 단일/다중 화자)
│   ├── callbacks.py       # Callback ABC + 5개 기본 콜백
│   ├── checkpoint.py      # CheckpointManager (meta.json 상태 관리)
│   ├── losses.py          # feature_loss, discriminator_loss, generator_loss, kl_loss
│   └── mel.py             # spectrogram_torch, mel_spectrogram_torch
├── inference/
│   └── synthesizer.py     # VITSSynthesizer (from_checkpoint, synthesize, ...)
└── cli/
    └── main.py            # vits train / infer / doctor / init / prepare
```

---

## 하위 호환성

원본 스크립트와 체크포인트는 그대로 사용할 수 있다.

```bash
# 원본 방식 (여전히 동작)
python train.py    -c trains/sample/config.json -m trains/sample/models
python train_ms.py -c trains/sample/config.json -m trains/sample/models

# 새 방식
vits train --config trains/sample/config.json
```

---

## 테스트

```bash
# 전체 단위 테스트
python -m pytest tests/unit/ --no-cov -q

# 커버리지 포함
python -m pytest tests/ --cov=src/vits --cov-report=term-missing
```

현재 **200개 테스트 모두 통과** (2026-03-04 기준).

---

## 구현 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| 0-1 | 프로젝트 인프라 (`pyproject.toml`, `Makefile`) | ✅ |
| 0-2 | 원본 코드 기반 테스트 확보 | ✅ |
| 0-3 | `src/vits/` 디렉토리 스캐폴딩 | ✅ |
| 1-1 | Pydantic v2 설정 스키마 | ✅ |
| 1-2 | 텍스트 플러그인 시스템 (`@register_language`) | ✅ |
| 1-3 | 데이터 파이프라인 | ✅ |
| 2-1 | 모델 코드 모듈 분리 | ✅ |
| 2-2 | 손실 함수 / Mel 처리 마이그레이션 | ✅ |
| 2-3 | `VITSTrainer` 클래스 | ✅ |
| 2-4 | 콜백 시스템 + Trainer 연동 | ✅ |
| 2-5 | `CheckpointManager` (meta.json) | ✅ |
| 3-1 | `VITSSynthesizer` 추론 클래스 | ✅ |
| 4-x | CLI (`vits` 명령어 5종) | ✅ |
| **5-1** | **코드 품질 (`ruff`, `mypy`)** | ❌ 미구현 |
| **5-2** | **패키징 (wheel 빌드, 배포)** | ❌ 미구현 |
| **5-3** | **API 문서 (`docs/`)** | ❌ 미구현 |

### 남은 작업 상세 (5-x)

#### 5-1: 코드 품질

```bash
# lint
ruff check src/vits/ --fix

# 타입 검사
mypy src/vits/ --ignore-missing-imports

# FutureWarning 수정: weight_norm → parametrizations.weight_norm
# 대상 파일: src/vits/models/generator.py, src/vits/models/discriminator.py
```

#### 5-2: 패키징

```bash
# wheel 빌드
pip install build
python -m build

# 배포 검증
pip install dist/vits_tts-0.1.0-py3-none-any.whl
vits doctor
```

`MANIFEST.in` 작성, `pyproject.toml`의 `version` / `classifiers` / `license` 보완 필요.

#### 5-3: API 문서

`src/vits/docs/` 하위에 가이드 작성 또는 `mkdocs` / `sphinx` 연동:
- 데이터 준비 가이드
- 언어 플러그인 추가 방법
- 콜백 커스텀 방법
- ONNX / TorchScript 내보내기 (`src/vits/export/`)

---

## 요구 사항

- Python ≥ 3.11
- PyTorch ≥ 2.0 (CUDA 12.x 권장)
- `monotonic_align` 빌드 (`Cython` 필요)
- 언어별 추가 패키지 (위 설치 섹션 참고)
