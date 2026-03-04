"""
유틸리티 함수 단위 테스트.
"""
import pytest
import sys
import os
import json
import tempfile
import numpy as np
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestHParams:
    """HParams 클래스 테스트."""

    def test_hparams_from_dict(self):
        """딕셔너리로 HParams 생성 가능한지."""
        from utils import HParams
        hp = HParams(a=1, b="hello", c={"nested": True})
        assert hp.a == 1
        assert hp.b == "hello"
        assert isinstance(hp.c, HParams)
        assert hp.c.nested is True

    def test_hparams_from_file(self, minimal_config_dict):
        """JSON 파일에서 HParams 로드."""
        from utils import get_hparams_from_file

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(minimal_config_dict, f)
            tmp_path = f.name

        try:
            hp = get_hparams_from_file(tmp_path)
            assert hp.train.epochs == minimal_config_dict["train"]["epochs"]
            assert hp.data.sampling_rate == 22050
        finally:
            os.unlink(tmp_path)

    def test_hparams_contains(self):
        """'in' 연산자 동작 확인."""
        from utils import HParams
        hp = HParams(foo=1, bar=2)
        assert "foo" in hp
        assert "baz" not in hp

    def test_hparams_iteration(self):
        """items() 동작 확인."""
        from utils import HParams
        hp = HParams(x=10, y=20)
        items = dict(hp.items())
        assert items["x"] == 10
        assert items["y"] == 20


class TestFileListUtils:
    """filelist 로드 유틸 테스트."""

    def test_load_filepaths_single_speaker(self):
        """싱글 화자 filelist 파싱 (path|text)."""
        from utils import load_filepaths_and_text

        content = "path/to/001.wav|안녕하세요.\npath/to/002.wav|반갑습니다.\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            result = load_filepaths_and_text(tmp_path)
            assert len(result) == 2
            assert result[0] == ["path/to/001.wav", "안녕하세요."]
            assert result[1] == ["path/to/002.wav", "반갑습니다."]
        finally:
            os.unlink(tmp_path)

    def test_load_filepaths_multi_speaker(self):
        """멀티 화자 filelist 파싱 (path|sid|text)."""
        from utils import load_filepaths_and_text

        content = "path/001.wav|0|안녕하세요.\npath/002.wav|1|반갑습니다.\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            result = load_filepaths_and_text(tmp_path)
            assert len(result) == 2
            assert result[0] == ["path/001.wav", "0", "안녕하세요."]
        finally:
            os.unlink(tmp_path)

    def test_load_filepaths_utf8(self):
        """UTF-8 인코딩 처리 확인."""
        from utils import load_filepaths_and_text

        # 다양한 유니코드 문자
        content = "a.wav|こんにちは\nb.wav|你好\nc.wav|مرحبا\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            result = load_filepaths_and_text(tmp_path)
            assert result[0][1] == "こんにちは"
            assert result[1][1] == "你好"
        finally:
            os.unlink(tmp_path)


class TestAudioUtils:
    """오디오 I/O 유틸 테스트."""

    def test_load_wav_to_torch(self, tmp_path):
        """WAV 파일을 torch Tensor로 로드."""
        from utils import load_wav_to_torch

        # 더미 WAV 파일 생성
        sr = 22050
        audio = (np.random.randn(sr) * 32767).astype(np.int16)
        wav_path = str(tmp_path / "test.wav")
        wavfile.write(wav_path, sr, audio)

        tensor, loaded_sr = load_wav_to_torch(wav_path)

        import torch
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert loaded_sr == sr
        assert tensor.shape == (sr,)

    def test_load_wav_float32(self, tmp_path):
        """로드된 tensor가 float32인지 확인."""
        from utils import load_wav_to_torch
        import torch

        sr = 16000
        audio = (np.random.randn(sr) * 32767).astype(np.int16)
        wav_path = str(tmp_path / "test16k.wav")
        wavfile.write(wav_path, sr, audio)

        tensor, _ = load_wav_to_torch(wav_path)
        assert tensor.dtype == torch.float32


class TestCheckpointUtils:
    """체크포인트 저장/로드 테스트."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """체크포인트 저장 후 로드가 동작하는지."""
        import torch
        from utils import save_checkpoint, load_checkpoint

        # 더미 모델
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ckpt_path = str(tmp_path / "G_1000.pth")
        save_checkpoint(model, optimizer, learning_rate=1e-3, iteration=1000, checkpoint_path=ckpt_path)

        assert os.path.exists(ckpt_path)

        # 새 모델에 로드
        model2 = torch.nn.Linear(10, 5)
        model2, _, lr, iteration = load_checkpoint(ckpt_path, model2)

        assert iteration == 1000
        assert abs(lr - 1e-3) < 1e-8

        # 가중치가 동일한지 확인
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_latest_checkpoint_path(self, tmp_path):
        """latest_checkpoint_path가 가장 큰 step 번호를 반환하는지."""
        from utils import latest_checkpoint_path
        import torch

        # 더미 체크포인트 파일 생성
        for step in [1000, 5000, 3000]:
            torch.save({}, str(tmp_path / f"G_{step}.pth"))

        latest = latest_checkpoint_path(str(tmp_path), "G_*.pth")
        assert "5000" in latest


class TestCommonsUtils:
    """commons.py 유틸 함수 테스트."""

    def test_sequence_mask_shape(self):
        """sequence_mask가 올바른 shape을 반환하는지."""
        import torch
        from commons import sequence_mask

        lengths = torch.LongTensor([5, 3, 7])
        mask = sequence_mask(lengths, max_length=10)

        assert mask.shape == (3, 10)
        assert mask[0].sum().item() == 5
        assert mask[1].sum().item() == 3
        assert mask[2].sum().item() == 7

    def test_rand_slice_segments_shape(self):
        """rand_slice_segments가 올바른 slice를 반환하는지."""
        import torch
        from commons import rand_slice_segments

        B, C, T = 4, 64, 200
        segment_size = 32
        x = torch.randn(B, C, T)
        x_lengths = torch.LongTensor([200, 180, 150, 120])

        sliced, ids = rand_slice_segments(x, x_lengths, segment_size)

        assert sliced.shape == (B, C, segment_size)
        assert ids.shape == (B,)

    def test_generate_path_sum(self):
        """generate_path의 각 열 합이 duration과 같은지."""
        import torch
        from commons import generate_path

        B, T_x = 1, 5
        duration = torch.LongTensor([[2, 3, 1, 4, 2]]).unsqueeze(1)  # [B, 1, T_x]
        T_y = duration.sum().item()
        mask = torch.ones(B, 1, T_y, T_x)

        path = generate_path(duration.float(), mask)

        # 각 텍스트 위치의 path 합이 duration과 같아야 함
        path_sums = path.squeeze(1).sum(1)  # [B, T_x]
        assert torch.allclose(path_sums.float(), duration.squeeze(1).float())
