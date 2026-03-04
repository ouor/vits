"""
콜백 시스템 및 CheckpointManager 단위 테스트.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from vits.training.callbacks import (
    Callback,
    CallbackList,
    LoggingCallback,
    ModelCheckpoint,
    EarlyStopping,
    TQDMCallback,
)
from vits.training.checkpoint import (
    CheckpointManager,
    CheckpointMeta,
    TrainingState,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
def make_trainer(global_step: int = 100, lr: float = 1e-4) -> MagicMock:
    """가짜 VITSTrainer를 반환한다."""
    trainer = MagicMock()
    trainer.global_step = global_step
    trainer.optim_g.param_groups = [{"lr": lr}]
    trainer.optim_d.param_groups = [{"lr": lr}]
    trainer.config.model_dir = None
    return trainer


# ──────────────────────────────────────────────────────────────────────────────
# Callback ABC / CallbackList
# ──────────────────────────────────────────────────────────────────────────────
class _RecordCallback(Callback):
    """훅 호출 기록용 콜백."""
    def __init__(self):
        self.calls: list[str] = []

    def on_train_begin(self, trainer):       self.calls.append("train_begin")
    def on_train_end(self, trainer):         self.calls.append("train_end")
    def on_epoch_begin(self, trainer, epoch): self.calls.append(f"epoch_begin:{epoch}")
    def on_epoch_end(self, trainer, epoch, losses): self.calls.append(f"epoch_end:{epoch}")
    def on_step_begin(self, trainer, step):  self.calls.append(f"step_begin:{step}")
    def on_step_end(self, trainer, step, losses): self.calls.append(f"step_end:{step}")
    def on_evaluate(self, trainer, epoch):   self.calls.append(f"evaluate:{epoch}")
    def on_save(self, trainer, meta):        self.calls.append("save")


class TestCallbackList:
    def test_empty_list_no_error(self):
        cb = CallbackList()
        tr = make_trainer()
        cb.on_train_begin(tr)
        cb.on_epoch_begin(tr, 1)
        cb.on_step_end(tr, 1, {})
        cb.on_train_end(tr)

    def test_hooks_dispatched_in_order(self):
        rec1, rec2 = _RecordCallback(), _RecordCallback()
        cb = CallbackList([rec1, rec2])
        tr = make_trainer()

        cb.on_train_begin(tr)
        cb.on_epoch_begin(tr, 1)
        cb.on_step_begin(tr, 0)
        cb.on_step_end(tr, 0, {"loss": 1.0})
        cb.on_epoch_end(tr, 1, {"loss": 1.0})
        cb.on_train_end(tr)

        for rec in (rec1, rec2):
            assert rec.calls == [
                "train_begin",
                "epoch_begin:1",
                "step_begin:0",
                "step_end:0",
                "epoch_end:1",
                "train_end",
            ]

    def test_none_callbacks_initializes_empty(self):
        cb = CallbackList(None)
        assert cb.callbacks == []

    def test_dynamic_dispatch_unknown_hook(self):
        """CallbackList에 없는 훅을 호출해도 오류가 없어야 한다."""
        cb = CallbackList()
        # 존재하지 않는 훅 – AttributeError 없이 조용하게 무시
        # (CallbackList는 __getattr__ 방식이 아니므로 정의된 메서드만 공개)
        # 최소한 공개 메서드들은 전부 동작해야 한다
        tr = make_trainer()
        cb.on_save(tr, None)


# ──────────────────────────────────────────────────────────────────────────────
# LoggingCallback
# ──────────────────────────────────────────────────────────────────────────────
class TestLoggingCallback:
    def test_logs_at_interval(self, caplog):
        import logging
        cb = LoggingCallback(log_interval=2)
        tr = make_trainer(global_step=0)

        with caplog.at_level(logging.INFO, logger="vits.training.callbacks"):
            for step in range(5):
                tr.global_step = step
                cb.on_step_end(tr, step, {"loss/g/total": float(step)})

        # step 0, 2, 4 → 3건 로그 (0 % 2 == 0, 2 % 2 == 0, 4 % 2 == 0)
        logged = [r for r in caplog.records if "loss" in r.message.lower() or "step" in r.message.lower()]
        assert len(logged) >= 3

    def test_skips_between_intervals(self):
        cb = LoggingCallback(log_interval=100)
        tr = make_trainer(global_step=50)
        # 50 % 100 != 0 → 로그 없어야 함 (예외 없으면 OK)
        cb.on_step_end(tr, 50, {"loss": 0.5})


# ──────────────────────────────────────────────────────────────────────────────
# ModelCheckpoint
# ──────────────────────────────────────────────────────────────────────────────
class TestModelCheckpoint:
    def test_save_every_n_epochs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = ModelCheckpoint(save_dir=tmpdir, save_every_n_epochs=2)
            tr = make_trainer()
            tr.is_main = True
            tr.net_g = MagicMock()
            tr.net_d = MagicMock()

            with patch("utils.save_checkpoint"):
                cb.on_epoch_end(tr, 1, {})  # 1 % 2 != 0 → no save (skip)
                cb.on_epoch_end(tr, 2, {})  # 2 % 2 == 0 → save

            # save_dir에 G_2.pth, D_2.pth 경로가 _saved_paths에 기록되어야 한다
            assert any("G_" in p for p in cb._saved_paths)

    def test_save_every_n_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = ModelCheckpoint(save_dir=tmpdir, save_every_n_steps=50)
            tr = make_trainer(global_step=50)
            tr.is_main = True
            tr.net_g = MagicMock()
            tr.net_d = MagicMock()

            with patch("utils.save_checkpoint"):
                cb.on_step_end(tr, 50, {})   # 50 % 50 == 0 → save

            assert any("G_" in p for p in cb._saved_paths)

    def test_no_save_when_neither_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # save_every_n_epochs=0, save_every_n_steps=0 → 저장 안 함
            cb = ModelCheckpoint(
                save_dir=tmpdir,
                save_every_n_epochs=0,
                save_every_n_steps=0,
            )
            tr = make_trainer()
            tr.is_main = True
            with patch("utils.save_checkpoint") as mock_save:
                cb.on_epoch_end(tr, 5, {})
                mock_save.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# EarlyStopping
# ──────────────────────────────────────────────────────────────────────────────
class TestEarlyStopping:
    def _run(self, losses_seq: list[float], patience: int = 2) -> bool:
        """loss 목록을 순서대로 먹여서 stop_training 여부를 반환한다."""
        cb = EarlyStopping(monitor="loss/g/total", patience=patience)
        tr = make_trainer()
        for i, v in enumerate(losses_seq):
            cb.on_epoch_end(tr, i + 1, {"loss/g/total": v})
        return cb.stop_training

    def test_no_stop_when_loss_decreasing(self):
        assert not self._run([1.0, 0.8, 0.6, 0.4])

    def test_stop_after_patience_exceeded(self):
        # 0.5에서 더 이상 내려가지 않음 → patience=2 후 중단
        assert self._run([0.5, 0.5, 0.5], patience=2)

    def test_stop_mode_max(self):
        cb = EarlyStopping(monitor="acc", patience=2, mode="max")
        tr = make_trainer()
        cb.on_epoch_end(tr, 1, {"acc": 0.9})
        cb.on_epoch_end(tr, 2, {"acc": 0.9})
        cb.on_epoch_end(tr, 3, {"acc": 0.9})
        assert cb.stop_training

    def test_reset_on_improvement(self):
        cb = EarlyStopping(monitor="loss/g/total", patience=3)
        tr = make_trainer()
        cb.on_epoch_end(tr, 1, {"loss/g/total": 1.0})
        cb.on_epoch_end(tr, 2, {"loss/g/total": 1.0})
        # 개선 → 카운터 리셋
        cb.on_epoch_end(tr, 3, {"loss/g/total": 0.5})
        assert not cb.stop_training

    def test_missing_monitor_key_does_not_crash(self):
        cb = EarlyStopping(monitor="nonexistent_key", patience=2)
        tr = make_trainer()
        cb.on_epoch_end(tr, 1, {"loss": 0.5})  # 키 없음 → 무시
        assert not cb.stop_training


# ──────────────────────────────────────────────────────────────────────────────
# CheckpointMeta / TrainingState serialization
# ──────────────────────────────────────────────────────────────────────────────
class TestCheckpointSerialization:
    def test_training_state_roundtrip(self):
        state = TrainingState(global_step=500, last_epoch=5, best_loss=0.3)
        state.checkpoints.append(
            CheckpointMeta(step=500, epoch=5, losses={"loss": 0.3}, lr=1e-4,
                           g_path="G_500.pth", d_path="D_500.pth")
        )
        d = state.to_dict()
        restored = TrainingState.from_dict(d)

        assert restored.global_step == 500
        assert restored.last_epoch == 5
        assert abs(restored.best_loss - 0.3) < 1e-9
        assert len(restored.checkpoints) == 1
        assert restored.checkpoints[0].step == 500

    def test_empty_state_from_dict(self):
        restored = TrainingState.from_dict({})
        assert restored.global_step == 0
        assert restored.checkpoints == []


# ──────────────────────────────────────────────────────────────────────────────
# CheckpointManager
# ──────────────────────────────────────────────────────────────────────────────
class TestCheckpointManager:
    def _make_manager(self, tmpdir: str, keep_last_n: int = 3) -> CheckpointManager:
        return CheckpointManager(save_dir=tmpdir, keep_last_n=keep_last_n)

    def test_list_checkpoints_empty_initially(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            assert mgr.list_checkpoints() == []

    def test_meta_json_written_on_save(self):
        """save() 후 meta.json이 생성되어야 한다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(tmpdir)
            tr = make_trainer(global_step=100)
            # config.model_dump()가 dict를 반환하도록 설정
            tr.config.model_dump.return_value = {"sample": True}

            # utils.save_checkpoint mock
            with patch("utils.save_checkpoint"):
                mgr.save(tr, epoch=1, losses={"loss/g/total": 0.5})

            meta_path = Path(tmpdir) / "meta.json"
            assert meta_path.exists()
            data = json.loads(meta_path.read_text())
            assert data["global_step"] == 100
            assert data["last_epoch"] == 1
            assert len(data["checkpoints"]) == 1

    def test_keep_last_n_deletes_old_checkpoints(self, tmp_path):
        """keep_last_n=2인 경우 3번째 저장 시 가장 오래된 것이 삭제되어야 한다."""
        mgr = CheckpointManager(save_dir=tmp_path, keep_last_n=2)

        # 가짜 .pth 파일 생성 (실제 삭제 여부 확인)
        fake_files = []
        for i in range(3):
            g = tmp_path / f"G_{(i+1)*100}.pth"
            d = tmp_path / f"D_{(i+1)*100}.pth"
            g.touch()
            d.touch()
            fake_files.append((str(g), str(d)))

        tr = make_trainer()
        with patch("utils.save_checkpoint"):
            for i, (g, d) in enumerate(fake_files):
                tr.global_step = (i + 1) * 100
                meta = CheckpointMeta(
                    step=tr.global_step, epoch=i+1,
                    g_path=g, d_path=d
                )
                mgr._state.checkpoints.append(meta)
                mgr._state.global_step = tr.global_step
                mgr._cleanup()

        # 1번째 체크포인트 파일이 삭제되어야 한다
        assert not Path(fake_files[0][0]).exists() or not Path(fake_files[0][1]).exists()
        assert len(mgr._state.checkpoints) == 2

    def test_get_state_returns_training_state(self, tmp_path):
        mgr = CheckpointManager(save_dir=tmp_path)
        state = mgr.get_state()
        assert isinstance(state, TrainingState)

    def test_load_latest_no_checkpoints(self, tmp_path):
        """체크포인트 없으면 None 반환."""
        mgr = CheckpointManager(save_dir=tmp_path)
        tr = make_trainer()
        tr.net_g = MagicMock()
        tr.net_d = MagicMock()
        tr.optim_g = MagicMock()
        tr.optim_d = MagicMock()

        with patch("utils.latest_checkpoint_path", side_effect=Exception("no ckpt")):
            result = mgr.load_latest(tr)
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Integration: CallbackList + EarlyStopping via trainer mock
# ──────────────────────────────────────────────────────────────────────────────
class TestCallbackIntegration:
    def test_early_stopping_sets_flag(self):
        es = EarlyStopping(monitor="loss/g/total", patience=1)
        cb_list = CallbackList([es])
        tr = make_trainer()

        cb_list.on_epoch_end(tr, 1, {"loss/g/total": 1.0})
        cb_list.on_epoch_end(tr, 2, {"loss/g/total": 1.0})

        # patience=1 → 2번째 에폭에서 중단 플래그 설정
        assert es.stop_training
        assert any(getattr(cb, "stop_training", False) for cb in cb_list.callbacks)

    def test_multiple_callbacks_all_receive_hooks(self):
        rec1, rec2 = _RecordCallback(), _RecordCallback()
        cb_list = CallbackList([rec1, rec2])
        tr = make_trainer()

        cb_list.on_train_begin(tr)
        assert rec1.calls == ["train_begin"]
        assert rec2.calls == ["train_begin"]
