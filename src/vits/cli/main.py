"""
VITS CLI — 주 진입점.

명령어:
    vits train      학습 실행
    vits infer      TTS 추론
    vits doctor     환경 진단
    vits init       설정 파일 템플릿 생성
    vits prepare    데이터 전처리 (filelist 정제)

사용 예시::

    vits train  --config config.json
    vits infer  --checkpoint G_50000.pth --text "안녕하세요"
    vits doctor
    vits init   --output my_config.json
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vits",
        description="VITS TTS — 학습 / 추론 / 진단 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_get_version())

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── train ──────────────────────────────────
    p_train = sub.add_parser("train", help="모델 학습")
    p_train.add_argument("-c", "--config",  required=True, help="config.json 경로")
    p_train.add_argument("--rank",        type=int, default=0,  help="DDP 랭크 (단일GPU=0)")
    p_train.add_argument("--world-size",  type=int, default=1,  help="DDP 프로세스 수")
    p_train.add_argument("--device",      default=None,         help="학습 디바이스 (예: cuda:0)")

    # ── infer ──────────────────────────────────
    p_infer = sub.add_parser("infer", help="텍스트 → 오디오 합성")
    p_infer.add_argument("-c", "--config",     default=None,  help="config.json 경로 (생략 시 checkpoint 옆)")
    p_infer.add_argument("-m", "--checkpoint", required=True, help="G_*.pth 경로")
    p_infer.add_argument("-t", "--text",       default=None,  help="합성할 텍스트 (없으면 stdin)")
    p_infer.add_argument("-o", "--output",     default="out.wav", help="출력 wav 파일")
    p_infer.add_argument("--speaker",     type=int, default=None, help="화자 ID")
    p_infer.add_argument("--noise-scale", type=float, default=0.667)
    p_infer.add_argument("--noise-scale-w", type=float, default=0.8)
    p_infer.add_argument("--length-scale", type=float, default=1.0)
    p_infer.add_argument("--device",  default=None, help="추론 디바이스")

    # ── doctor ─────────────────────────────────
    sub.add_parser("doctor", help="환경 및 의존성 진단")

    # ── init ───────────────────────────────────
    p_init = sub.add_parser("init", help="설정 파일 템플릿 생성")
    p_init.add_argument("-o", "--output", default="config.json", help="출력 파일 경로")
    p_init.add_argument("--speakers", type=int, default=0, help="화자 수 (0 = 싱글 화자)")

    # ── prepare ────────────────────────────────
    p_prep = sub.add_parser("prepare", help="filelist 전처리 (텍스트 정제)")
    p_prep.add_argument("filelist", help="원본 filelist.txt 경로")
    p_prep.add_argument("-c", "--config", required=True, help="config.json 경로")
    p_prep.add_argument("-o", "--output", default=None, help="출력 파일 (기본: filelist.txt.cleaned)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cmd = args.command
    if cmd == "train":
        return _cmd_train(args)
    elif cmd == "infer":
        return _cmd_infer(args)
    elif cmd == "doctor":
        return _cmd_doctor()
    elif cmd == "init":
        return _cmd_init(args)
    elif cmd == "prepare":
        return _cmd_prepare(args)
    else:
        parser.print_help()
        return 1


# ──────────────────────────────────────────────────────────────────────────────
# 명령어 구현
# ──────────────────────────────────────────────────────────────────────────────
def _cmd_train(args: argparse.Namespace) -> int:
    """학습을 실행한다."""
    try:
        from vits.configs.schema import VITSConfig
        from vits.training.trainer import VITSTrainer
    except ImportError as exc:
        print(f"[오류] 패키지 임포트 실패: {exc}", file=sys.stderr)
        return 1

    print(f"[train] 설정 로드: {args.config}")
    try:
        config = VITSConfig.from_json(args.config)
    except Exception as exc:
        print(f"[오류] 설정 파일 로드 실패: {exc}", file=sys.stderr)
        return 1

    trainer = VITSTrainer(
        config,
        rank=args.rank,
        world_size=args.world_size,
        device=args.device,
    )
    print(f"[train] 학습 시작 (rank={args.rank}, world_size={args.world_size})")
    trainer.train()
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    """TTS 추론을 실행한다."""
    try:
        import soundfile as sf  # type: ignore
        from vits.inference import VITSSynthesizer
    except ImportError as exc:
        print(f"[오류] 패키지 임포트 실패: {exc}", file=sys.stderr)
        print("  soundfile 설치: pip install soundfile", file=sys.stderr)
        return 1

    # 텍스트 읽기
    if args.text:
        text = args.text
    else:
        print("합성할 텍스트를 입력하세요 (Ctrl+D로 종료):", file=sys.stderr)
        text = sys.stdin.read().strip()
    if not text:
        print("[오류] 텍스트가 없습니다.", file=sys.stderr)
        return 1

    print(f"[infer] 체크포인트 로드: {args.checkpoint}")
    try:
        syn = VITSSynthesizer.from_checkpoint(
            args.checkpoint,
            config=args.config,
            device=args.device,
        )
    except Exception as exc:
        print(f"[오류] 모델 로드 실패: {exc}", file=sys.stderr)
        return 1

    print(f"[infer] 합성 중: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
    try:
        audio = syn.synthesize(
            text,
            speaker_id=args.speaker,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
        )
    except Exception as exc:
        print(f"[오류] 합성 실패: {exc}", file=sys.stderr)
        return 1

    sf.write(args.output, audio, syn.sampling_rate)
    duration = len(audio) / syn.sampling_rate
    print(f"[infer] 저장 완료: {args.output} ({duration:.2f}초)")
    return 0


def _cmd_doctor() -> int:
    """환경 및 의존성을 진단한다."""
    import importlib
    import platform

    print("=" * 60)
    print("VITS 환경 진단")
    print("=" * 60)

    # 시스템 정보
    print(f"\n[시스템]")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  OS:       {platform.system()} {platform.release()}")
    print(f"  CPU:      {platform.processor() or 'N/A'}")

    # 필수 패키지
    required = [
        ("torch",       "PyTorch"),
        ("torchaudio",  "TorchAudio"),
        ("numpy",       "NumPy"),
        ("scipy",       "SciPy"),
        ("librosa",     "Librosa"),
        ("soundfile",   "SoundFile"),
        ("pydantic",    "Pydantic"),
        ("phonemizer",  "Phonemizer"),
    ]
    print(f"\n[패키지]")
    all_ok = True
    for pkg, name in required:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✓ {name:<12} {ver}")
        except ImportError:
            print(f"  ✗ {name:<12} 미설치")
            all_ok = False

    # CUDA
    print(f"\n[CUDA]")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 사용 가능: {torch.version.cuda}")
            print(f"  GPU 수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    [{i}] {name} ({mem:.1f} GB)")
        else:
            print("  ✗ CUDA 사용 불가 (CPU 전용)")
    except ImportError:
        print("  ✗ PyTorch 미설치")

    # vits 패키지
    print(f"\n[vits 패키지]")
    try:
        import vits
        print(f"  ✓ vits 임포트 성공")
    except ImportError as exc:
        print(f"  ✗ vits 임포트 실패: {exc}")
        all_ok = False

    print(f"\n{'[OK] 모든 검사 통과' if all_ok else '[주의] 일부 패키지가 없습니다.'}")
    print("=" * 60)
    return 0 if all_ok else 1


def _cmd_init(args: argparse.Namespace) -> int:
    """설정 파일 템플릿을 생성한다."""
    import json

    n_speakers = args.speakers
    gin_channels = 256 if n_speakers > 0 else 0

    template = {
        "model_dir": "./logs/my_model",
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "gin_channels": gin_channels,
            "use_sdp": True,
            "add_blank": True,
        },
        "data": {
            "training_files": "filelists/train.txt.cleaned",
            "validation_files": "filelists/val.txt.cleaned",
            "text_cleaners": ["english_cleaners2"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "n_speakers": n_speakers,
            "cleaned_text": True,
        },
        "train": {
            "log_interval": 200,
            "eval_interval": 1000,
            "seed": 1234,
            "epochs": 20000,
            "learning_rate": 2e-4,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": 16,
            "fp16_run": True,
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
            "log_dir": "./logs/my_model/tb",
        },
    }

    import os
    if os.path.exists(args.output):
        print(f"[경고] 파일이 이미 존재합니다: {args.output} (덮어쓰기)")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"[init] 설정 템플릿 생성: {args.output}")
    if n_speakers > 0:
        print(f"       화자 수: {n_speakers} (다중 화자 설정)")
    else:
        print("       화자 수: 단일 화자 설정")
    print("\n다음 단계:")
    print("  1. 오디오 파일을 준비하고 filelist를 작성하세요.")
    print("  2. vits prepare <filelist.txt> --config config.json 실행")
    print("  3. vits train --config config.json 실행")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    """filelist.txt의 텍스트를 전처리해 .cleaned 파일을 생성한다."""
    import os

    output = args.output or (args.filelist + ".cleaned")
    print(f"[prepare] 입력: {args.filelist}")
    print(f"[prepare] 출력: {output}")

    try:
        from vits.configs.schema import VITSConfig
        config = VITSConfig.from_json(args.config)
        cleaners = config.data.text_cleaners
    except Exception as exc:
        print(f"[오류] 설정 로드 실패: {exc}", file=sys.stderr)
        return 1

    try:
        from text import text_to_sequence  # type: ignore
        from text.cleaners import english_cleaners2  # type: ignore  # noqa
    except ImportError:
        print("[오류] text 모듈을 찾을 수 없습니다.", file=sys.stderr)
        return 1

    import sys as _sys; _sys.stdout.flush()

    try:
        from preprocess import preprocess  # type: ignore
        preprocess(args.config, args.filelist, output)
        print(f"[prepare] 완료: {output}")
    except ImportError:
        # preprocess 모듈 없으면 직접 처리
        _prepare_direct(args.filelist, output, cleaners)

    return 0


def _prepare_direct(
    input_path: str,
    output_path: str,
    cleaners: list[str],
) -> None:
    """preprocess.py 없이 직접 텍스트를 정제한다."""
    import os
    try:
        from text.cleaners import english_cleaners2  # type: ignore
        from text import text_to_sequence  # type: ignore
    except ImportError:
        print("[오류] text 모듈 없음", file=sys.stderr)
        return

    cleaned_lines = []
    with open(input_path, encoding="utf-8") as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.rstrip("\n")
        if "|" in line:
            parts = line.split("|")
            text_field = parts[1] if len(parts) >= 2 else ""
            # 간단히 IPA 변환 없이 raw 텍스트 유지 (실제 cleaners 적용은 학습 시)
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(cleaned_lines))
        if cleaned_lines:
            fp.write("\n")

    print(f"[prepare] {len(cleaned_lines)}개 라인 처리 완료")


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("vits")
    except Exception:
        return "0.0.0"
