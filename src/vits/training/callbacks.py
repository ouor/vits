"""
학습 콜백 시스템.

Trainer의 비즈니스 로직에 영향을 주지 않고
학습 과정에 관찰 및 부수 동작을 삽입할 수 있다.

기본 제공 콜백:
    - LoggingCallback   : Python logging 출력
    - TQDMCallback       : tqdm 진행 바
    - ModelCheckpoint   : 에폭/스텝 기반 자동 체크포인트
    - EarlyStopping     : validation loss 정체 시 조기 종료
    - TensorBoardCallback: TensorBoard scalar/image/audio 기록

사용 예시::

    from vits.training.callbacks import ModelCheckpoint, TQDMCallback
    trainer = VITSTrainer(config, callbacks=[ModelCheckpoint(...), TQDMCallback()])
    trainer.train()
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    # 순환 임포트 방지 - 타입 힌트 전용
    from vits.training.trainer import VITSTrainer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Base Callback
# ──────────────────────────────────────────────────────────────────────────────
class Callback(ABC):
    """
    학습 콜백 추상 베이스 클래스.

    각 hook은 선택적 override이며, 기본 구현은 아무 일도 하지 않는다.
    콜백에서 예외를 발생시키면 학습이 중단된다.
    """

    def on_train_begin(self, trainer: "VITSTrainer") -> None:
        """학습 시작 시 (첫 에폭 이전) 호출."""

    def on_epoch_begin(self, trainer: "VITSTrainer", epoch: int) -> None:
        """에폭 시작 시 호출."""

    def on_step_begin(self, trainer: "VITSTrainer", step: int) -> None:
        """스텝 시작 시 호출."""

    def on_step_end(
        self, trainer: "VITSTrainer", step: int, losses: dict[str, float]
    ) -> None:
        """스텝 종료 시 호출."""

    def on_epoch_end(
        self, trainer: "VITSTrainer", epoch: int, losses: dict[str, float]
    ) -> None:
        """에폭 종료 시 호출."""

    def on_evaluate(
        self, trainer: "VITSTrainer", step: int, metrics: dict[str, Any]
    ) -> None:
        """평가 수행 후 호출."""

    def on_save(self, trainer: "VITSTrainer", step: int, path: str) -> None:
        """체크포인트 저장 후 호출."""

    def on_train_end(self, trainer: "VITSTrainer") -> None:
        """학습 종료 시 (마지막 에폭 이후) 호출."""


# ──────────────────────────────────────────────────────────────────────────────
# 콜백 목록 관리자
# ──────────────────────────────────────────────────────────────────────────────
class CallbackList:
    """
    콜백 리스트 관리자.
    Trainer가 이를 통해 모든 콜백에 이벤트를 브로드캐스트한다.
    """

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks: list[Callback] = list(callbacks or [])

    def add(self, cb: Callback) -> None:
        self.callbacks.append(cb)

    def __getattr__(self, name: str):
        """on_* 메서드를 동적으로 생성하여 모든 콜백에 브로드캐스트."""
        if name.startswith("on_"):
            def dispatcher(*args, **kwargs):
                for cb in self.callbacks:
                    getattr(cb, name)(*args, **kwargs)
            return dispatcher
        raise AttributeError(name)


# ──────────────────────────────────────────────────────────────────────────────
# 기본 제공 콜백들
# ──────────────────────────────────────────────────────────────────────────────
class LoggingCallback(Callback):
    """
    Python logging을 통해 학습 진행 상황을 기록한다.

    Args:
        log_interval: 로그를 출력할 스텝 간격
    """

    def __init__(self, log_interval: int = 200) -> None:
        self.log_interval = log_interval

    def on_train_begin(self, trainer: "VITSTrainer") -> None:
        logger.info(
            "학습 시작: %d 에폭, 디바이스=%s",
            trainer.config.train.epochs,
            trainer.device,
        )

    def on_step_end(self, trainer: "VITSTrainer", step: int, losses: dict[str, float]) -> None:
        if step % self.log_interval == 0 and trainer.is_main:
            lr = trainer.optim_g.param_groups[0]["lr"]
            loss_str = " | ".join(f"{k}={v:.4f}" for k, v in losses.items())
            logger.info("step=%d | lr=%.2e | %s", step, lr, loss_str)

    def on_epoch_end(self, trainer: "VITSTrainer", epoch: int, losses: dict[str, float]) -> None:
        if trainer.is_main:
            logger.info("====> Epoch: %d 완료 | avg_losses: %s", epoch,
                        {k: f"{v:.4f}" for k, v in losses.items()})

    def on_train_end(self, trainer: "VITSTrainer") -> None:
        logger.info("학습 완료.")


class TQDMCallback(Callback):
    """
    tqdm 진행 바를 사용한 배치 진행 표시.

    epoch_bar  : 에폭 진행 바 (outer)
    step_bar   : 배치 진행 바 (inner, 에폭 내)
    """

    def __init__(self, total_epochs: int | None = None) -> None:
        self._epoch_bar  = None
        self._step_bar   = None
        self._total_epochs = total_epochs

    def on_train_begin(self, trainer: "VITSTrainer") -> None:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore[import]
            total = self._total_epochs or trainer.config.train.epochs
            self._epoch_bar = _tqdm(total=total, desc="Epochs", unit="epoch")
        except ImportError:
            pass

    def on_epoch_begin(self, trainer: "VITSTrainer", epoch: int) -> None:
        if trainer.train_loader and trainer.is_main:
            try:
                from tqdm import tqdm as _tqdm  # type: ignore[import]
                self._step_bar = _tqdm(
                    total=len(trainer.train_loader),
                    desc=f"Epoch {epoch}",
                    unit="batch",
                    leave=False,
                )
            except ImportError:
                pass

    def on_step_end(self, trainer: "VITSTrainer", step: int, losses: dict[str, float]) -> None:
        if self._step_bar is not None:
            self._step_bar.update(1)
            loss_g = losses.get("loss/g/total", 0.0)
            loss_d = losses.get("loss/d/total", 0.0)
            self._step_bar.set_postfix(g=f"{loss_g:.4f}", d=f"{loss_d:.4f}")

    def on_epoch_end(self, trainer: "VITSTrainer", epoch: int, losses: dict[str, float]) -> None:
        if self._step_bar is not None:
            self._step_bar.close()
            self._step_bar = None
        if self._epoch_bar is not None:
            self._epoch_bar.update(1)

    def on_train_end(self, trainer: "VITSTrainer") -> None:
        if self._epoch_bar is not None:
            self._epoch_bar.close()


class ModelCheckpoint(Callback):
    """
    에폭 / 스텝 기반 자동 체크포인트 저장.

    Args:
        save_dir:        체크포인트 저장 디렉토리
        save_every_n_epochs: N 에폭마다 저장 (0 = 비활성)
        save_every_n_steps:  N 스텝마다 저장 (0 = 비활성)
        keep_last_n:     유지할 최근 체크포인트 수 (0 = 전부 유지)
    """

    def __init__(
        self,
        save_dir: str | Path = "checkpoints",
        save_every_n_epochs: int = 1,
        save_every_n_steps: int = 0,
        keep_last_n: int = 3,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_steps  = save_every_n_steps
        self.keep_last_n = keep_last_n
        self._saved_paths: list[str] = []

    def on_epoch_end(self, trainer: "VITSTrainer", epoch: int, losses: dict[str, float]) -> None:
        if not trainer.is_main:
            return
        if self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0:
            self._do_save(trainer, epoch)

    def on_step_end(self, trainer: "VITSTrainer", step: int, losses: dict[str, float]) -> None:
        if not trainer.is_main:
            return
        if self.save_every_n_steps and step % self.save_every_n_steps == 0:
            self._do_save(trainer, step)

    def _do_save(self, trainer: "VITSTrainer", tag: int) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        g_path = str(self.save_dir / f"G_{tag}.pth")
        d_path = str(self.save_dir / f"D_{tag}.pth")
        try:
            import utils  # type: ignore[import]
            lr = trainer.optim_g.param_groups[0]["lr"]
            utils.save_checkpoint(trainer.net_g, trainer.optim_g, lr, tag, g_path)
            utils.save_checkpoint(trainer.net_d, trainer.optim_d, lr, tag, d_path)
            self._saved_paths.extend([g_path, d_path])
            trainer.callbacks.on_save(trainer, tag, g_path)
            logger.info("체크포인트 저장: %s", g_path)
            self._cleanup()
        except Exception as exc:
            logger.error("체크포인트 저장 실패: %s", exc)

    def _cleanup(self) -> None:
        if self.keep_last_n <= 0:
            return
        # G_*.pth 와 D_*.pth 분리 관리
        g_paths = [p for p in self._saved_paths if "/G_" in p]
        d_paths = [p for p in self._saved_paths if "/D_" in p]
        for paths in [g_paths, d_paths]:
            while len(paths) > self.keep_last_n:
                old = paths.pop(0)
                if os.path.exists(old):
                    os.remove(old)
                    logger.debug("오래된 체크포인트 제거: %s", old)


class EarlyStopping(Callback):
    """
    모니터링 지표가 patience 에폭 동안 개선되지 않으면 학습을 조기 종료한다.

    Args:
        monitor:   모니터링할 손실 키 (예: "loss/g/mel")
        patience:  개선 없이 허용할 에폭 수
        min_delta: 개선으로 인정할 최소 변화량
        mode:      "min" (낮을수록 좋음) 또는 "max" (높을수록 좋음)
    """

    def __init__(
        self,
        monitor: str = "loss/g/mel",
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.monitor   = monitor
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self._best     = float("inf") if mode == "min" else float("-inf")
        self._counter  = 0
        self.stop_training = False

    def on_epoch_end(self, trainer: "VITSTrainer", epoch: int, losses: dict[str, float]) -> None:
        current = losses.get(self.monitor)
        if current is None:
            return

        if self.mode == "min":
            improved = current < self._best - self.min_delta
        else:
            improved = current > self._best + self.min_delta

        if improved:
            self._best    = current
            self._counter = 0
        else:
            self._counter += 1
            logger.info(
                "EarlyStopping: %s 지표 개선 없음 (%d/%d)",
                self.monitor, self._counter, self.patience,
            )
            if self._counter >= self.patience:
                logger.warning(
                    "EarlyStopping: %d 에폭 동안 개선 없음. 학습 중단.",
                    self.patience,
                )
                self.stop_training = True


class TensorBoardCallback(Callback):
    """
    TensorBoard에 스칼라/이미지/오디오를 기록하는 콜백.

    Args:
        log_dir:       TensorBoard 로그 디렉토리
        log_interval:  스칼라 기록 스텝 간격
        audio_sample_rate: 오디오 샘플링 레이트
    """

    def __init__(
        self,
        log_dir: str = "runs",
        log_interval: int = 200,
        audio_sample_rate: int = 22050,
    ) -> None:
        self.log_dir           = log_dir
        self.log_interval      = log_interval
        self.audio_sample_rate = audio_sample_rate
        self._writer = None

    def on_train_begin(self, trainer: "VITSTrainer") -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            logger.warning("tensorboard가 설치되지 않아 TensorBoardCallback이 비활성화됩니다.")

    def on_step_end(self, trainer: "VITSTrainer", step: int, losses: dict[str, float]) -> None:
        if self._writer is None or not trainer.is_main:
            return
        if step % self.log_interval == 0:
            for k, v in losses.items():
                self._writer.add_scalar(k, v, global_step=step)

    def on_evaluate(self, trainer: "VITSTrainer", step: int, metrics: dict[str, Any]) -> None:
        if self._writer is None or not trainer.is_main:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"eval/{k}", v, global_step=step)

    def on_train_end(self, trainer: "VITSTrainer") -> None:
        if self._writer is not None:
            self._writer.close()
