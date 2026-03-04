"""
체크포인트 상태 관리자.

학습 상태(step, epoch, loss 히스토리, 설정 스냅샷)를
하나의 메타데이터 파일로 관리한다.

저장 구조::

    checkpoints/
        G_5000.pth          ← 생성기 가중치 + 옵티마이저
        D_5000.pth          ← 판별기 가중치 + 옵티마이저
        meta.json           ← 학습 상태 (global_step, epoch, loss 히스토리 등)

사용 예시::

    mgr = CheckpointManager("checkpoints", keep_last_n=3)
    mgr.save(trainer, epoch=5, step=5000, losses={...})
    info = mgr.load_latest(trainer)
    print(info.epoch, info.step)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vits.training.trainer import VITSTrainer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 체크포인트 메타데이터
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CheckpointMeta:
    """단일 체크포인트의 메타데이터."""
    step:      int
    epoch:     int
    losses:    dict[str, float] = field(default_factory=dict)
    lr:        float = 0.0
    g_path:    str = ""
    d_path:    str = ""
    extra:     dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingState:
    """전체 학습 상태 (meta.json에 저장)."""
    global_step:    int = 0
    last_epoch:     int = 0
    best_loss:      float = float("inf")
    checkpoints:    list[CheckpointMeta] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingState":
        state = cls(
            global_step=d.get("global_step", 0),
            last_epoch=d.get("last_epoch", 0),
            best_loss=d.get("best_loss", float("inf")),
            config_snapshot=d.get("config_snapshot", {}),
        )
        for cm in d.get("checkpoints", []):
            state.checkpoints.append(CheckpointMeta(**cm))
        return state


# ──────────────────────────────────────────────────────────────────────────────
# CheckpointManager
# ──────────────────────────────────────────────────────────────────────────────
class CheckpointManager:
    """
    체크포인트 저장/복원/관리.

    Args:
        save_dir:    체크포인트 저장 디렉토리
        keep_last_n: 유지할 최근 체크포인트 수 (0 = 전부 유지)
    """

    META_FILE = "meta.json"

    def __init__(
        self,
        save_dir: str | Path = "checkpoints",
        keep_last_n: int = 3,
    ) -> None:
        self.save_dir    = Path(save_dir)
        self.keep_last_n = keep_last_n
        self._state      = TrainingState()

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def save(
        self,
        trainer: "VITSTrainer",
        epoch: int,
        losses: dict[str, float] | None = None,
    ) -> CheckpointMeta:
        """
        모델 + 옵티마이저 + 학습 상태를 저장한다.

        Returns:
            저장된 체크포인트의 메타데이터
        """
        import utils  # type: ignore[import]

        self.save_dir.mkdir(parents=True, exist_ok=True)
        step = trainer.global_step
        lr   = trainer.optim_g.param_groups[0]["lr"]

        g_path = str(self.save_dir / f"G_{step}.pth")
        d_path = str(self.save_dir / f"D_{step}.pth")

        utils.save_checkpoint(trainer.net_g, trainer.optim_g, lr, epoch, g_path)
        utils.save_checkpoint(trainer.net_d, trainer.optim_d, lr, epoch, d_path)

        meta = CheckpointMeta(
            step=step,
            epoch=epoch,
            losses=losses or {},
            lr=lr,
            g_path=g_path,
            d_path=d_path,
        )
        self._state.global_step  = step
        self._state.last_epoch   = epoch
        self._state.checkpoints.append(meta)

        # 손실 기록
        total_loss = losses.get("loss/g/total", float("inf")) if losses else float("inf")
        if total_loss < self._state.best_loss:
            self._state.best_loss = total_loss

        self._save_meta(trainer)
        self._cleanup()
        logger.info("체크포인트 저장: step=%d, epoch=%d → %s", step, epoch, g_path)
        return meta

    def load_latest(self, trainer: "VITSTrainer") -> Optional[CheckpointMeta]:
        """
        가장 최근 체크포인트를 로드하고 메타데이터를 반환한다.
        체크포인트가 없으면 None을 반환한다.
        """
        import utils  # type: ignore[import]

        meta_path = self.save_dir / self.META_FILE
        if meta_path.exists():
            self._state = TrainingState.from_dict(
                json.loads(meta_path.read_text(encoding="utf-8"))
            )

        if not self._state.checkpoints:
            # meta.json이 없거나 비어있는 경우 — 파일 시스템에서 탐색
            return self._load_from_filesystem(trainer)

        latest = self._state.checkpoints[-1]
        try:
            _, _, _, epoch = utils.load_checkpoint(latest.g_path, trainer.net_g, trainer.optim_g)
            _, _, _, _     = utils.load_checkpoint(latest.d_path, trainer.net_d, trainer.optim_d)
            trainer.global_step = latest.step
            logger.info("체크포인트 로드: step=%d, epoch=%d", latest.step, epoch)
            return latest
        except Exception as exc:
            logger.warning("체크포인트 로드 실패: %s", exc)
            return None

    def get_state(self) -> TrainingState:
        """현재 학습 상태를 반환한다."""
        return self._state

    def list_checkpoints(self) -> list[CheckpointMeta]:
        """저장된 모든 체크포인트 메타데이터 목록을 반환한다."""
        return list(self._state.checkpoints)

    # ── 내부 구현 ─────────────────────────────────────────────────────────────

    def _save_meta(self, trainer: "VITSTrainer") -> None:
        """TrainingState를 meta.json으로 직렬화한다."""
        # 설정 스냅샷 업데이트
        try:
            self._state.config_snapshot = trainer.config.model_dump()
        except Exception:
            pass

        meta_path = self.save_dir / self.META_FILE
        meta_path.write_text(
            json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cleanup(self) -> None:
        """keep_last_n 초과 체크포인트를 삭제한다."""
        if self.keep_last_n <= 0:
            return
        while len(self._state.checkpoints) > self.keep_last_n:
            old = self._state.checkpoints.pop(0)
            for path in (old.g_path, old.d_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.debug("오래된 체크포인트 제거: %s", path)
                    except OSError as exc:
                        logger.warning("파일 삭제 실패: %s (%s)", path, exc)

    def _load_from_filesystem(self, trainer: "VITSTrainer") -> Optional[CheckpointMeta]:
        """meta.json 없이 파일 시스템에서 최근 체크포인트를 탐색한다 (하위 호환)."""
        import utils  # type: ignore[import]

        try:
            g_path = utils.latest_checkpoint_path(str(self.save_dir), "G_*.pth")
            d_path = utils.latest_checkpoint_path(str(self.save_dir), "D_*.pth")
            _, _, lr, epoch = utils.load_checkpoint(g_path, trainer.net_g, trainer.optim_g)
            _, _, _, _      = utils.load_checkpoint(d_path, trainer.net_d, trainer.optim_d)

            # step 추정: 파일명에서 숫자 추출
            step = int(Path(g_path).stem.split("_")[1]) if "_" in Path(g_path).stem else 0
            trainer.global_step = step

            meta = CheckpointMeta(step=step, epoch=epoch, lr=lr, g_path=g_path, d_path=d_path)
            self._state.checkpoints.append(meta)
            self._state.global_step = step
            self._state.last_epoch  = epoch
            logger.info("파일시스템 체크포인트 로드: %s (epoch=%d)", g_path, epoch)
            return meta
        except Exception:
            logger.info("저장된 체크포인트가 없습니다. 처음부터 학습합니다.")
            return None
