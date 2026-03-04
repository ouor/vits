"""
CLI 단위 테스트.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from vits.cli.main import build_parser, main


# ──────────────────────────────────────────────────────────────────────────────
# 파서 구조 테스트
# ──────────────────────────────────────────────────────────────────────────────
class TestParser:
    def test_build_parser_returns_parser(self):
        p = build_parser()
        assert p is not None

    def test_help_does_not_raise(self):
        p = build_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--help"])
        assert exc.value.code == 0

    def test_version_does_not_raise(self):
        p = build_parser()
        with pytest.raises(SystemExit) as exc:
            p.parse_args(["--version"])
        assert exc.value.code == 0

    def test_subcommand_required(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_train_requires_config(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["train"])

    def test_train_parses_correctly(self):
        p = build_parser()
        args = p.parse_args(["train", "--config", "cfg.json"])
        assert args.command == "train"
        assert args.config == "cfg.json"
        assert args.rank == 0
        assert args.world_size == 1

    def test_infer_requires_checkpoint(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["infer"])

    def test_infer_parses_correctly(self):
        p = build_parser()
        args = p.parse_args([
            "infer",
            "--checkpoint", "G_5000.pth",
            "--text", "hello",
            "--output", "out.wav",
        ])
        assert args.command == "infer"
        assert args.checkpoint == "G_5000.pth"
        assert args.text == "hello"
        assert args.output == "out.wav"

    def test_init_defaults(self):
        p = build_parser()
        args = p.parse_args(["init"])
        assert args.output == "config.json"
        assert args.speakers == 0


# ──────────────────────────────────────────────────────────────────────────────
# doctor 명령어
# ──────────────────────────────────────────────────────────────────────────────
class TestDoctor:
    def test_doctor_returns_int(self, capsys):
        result = main(["doctor"])
        assert result in (0, 1)

    def test_doctor_prints_header(self, capsys):
        main(["doctor"])
        out = capsys.readouterr().out
        assert "VITS" in out or "Python" in out


# ──────────────────────────────────────────────────────────────────────────────
# init 명령어
# ──────────────────────────────────────────────────────────────────────────────
class TestInit:
    def test_init_creates_json(self, tmp_path):
        out = str(tmp_path / "config.json")
        rc = main(["init", "--output", out])
        assert rc == 0
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert "model" in data
        assert "data" in data
        assert "train" in data

    def test_init_single_speaker(self, tmp_path):
        out = str(tmp_path / "config.json")
        main(["init", "--output", out, "--speakers", "0"])
        data = json.loads(Path(out).read_text())
        assert data["data"]["n_speakers"] == 0
        assert data["model"]["gin_channels"] == 0

    def test_init_multi_speaker(self, tmp_path):
        out = str(tmp_path / "config.json")
        main(["init", "--output", out, "--speakers", "4"])
        data = json.loads(Path(out).read_text())
        assert data["data"]["n_speakers"] == 4
        assert data["model"]["gin_channels"] == 256

    def test_init_valid_vitsconfig(self, tmp_path):
        """생성된 JSON이 VITSConfig로 파싱 가능한지 확인한다."""
        from vits.configs.schema import VITSConfig
        out = str(tmp_path / "config.json")
        main(["init", "--output", out])
        cfg = VITSConfig.from_json(out)
        assert cfg.train.epochs == 20000


# ──────────────────────────────────────────────────────────────────────────────
# train 명령어 (mock)
# ──────────────────────────────────────────────────────────────────────────────
class TestTrain:
    def test_train_with_valid_config(self, tmp_path):
        """VITSConfig가 로드되면 VITSTrainer가 생성/호출되어야 한다."""
        out = str(tmp_path / "config.json")
        main(["init", "--output", out])

        mock_trainer = MagicMock()
        with patch("vits.training.trainer.VITSTrainer", return_value=mock_trainer) as MockClass:
            rc = main(["train", "--config", out])
        assert rc == 0
        mock_trainer.train.assert_called_once()

    def test_train_with_missing_config(self, tmp_path):
        rc = main(["train", "--config", str(tmp_path / "nonexistent.json")])
        assert rc != 0


# ──────────────────────────────────────────────────────────────────────────────
# infer 명령어 (mock)
# ──────────────────────────────────────────────────────────────────────────────
class TestInfer:
    def test_infer_mock(self, tmp_path):
        import numpy as np

        mock_syn = MagicMock()
        mock_syn.synthesize.return_value = np.zeros(22050, dtype=np.float32)
        mock_syn.sampling_rate = 22050

        ckpt = str(tmp_path / "G_100.pth")
        Path(ckpt).touch()
        out_wav = str(tmp_path / "out.wav")

        with patch("vits.inference.synthesizer.VITSSynthesizer.from_checkpoint",
                   return_value=mock_syn):
            with patch("soundfile.write") as mock_write:
                rc = main([
                    "infer",
                    "--checkpoint", ckpt,
                    "--text", "hello",
                    "--output", out_wav,
                ])
        assert rc == 0
        mock_syn.synthesize.assert_called_once_with(
            "hello",
            speaker_id=None,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0,
        )

    def test_infer_no_text_uses_empty_error(self, tmp_path):
        ckpt = str(tmp_path / "G_100.pth")
        Path(ckpt).touch()
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = ""  # 빈 stdin
            rc = main([
                "infer",
                "--checkpoint", ckpt,
                "--output", str(tmp_path / "out.wav"),
            ])
        assert rc != 0  # 빈 텍스트 → 오류
