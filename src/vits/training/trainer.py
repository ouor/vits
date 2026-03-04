"""
VITS 학습 엔진.

단일 화자 / 다중 화자, 단일 GPU / 분산 학습(DDP)을 하나의 클래스로 통합한다.

기본 사용::

    from vits.configs import VITSConfig
    from vits.training.trainer import VITSTrainer

    config = VITSConfig.from_json("config.json")
    trainer = VITSTrainer(config)
    trainer.train()

DDP (torchrun)::

    # 각 프로세스에서
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    trainer = VITSTrainer(config, rank=rank, world_size=dist.get_world_size())
    trainer.train()
    dist.destroy_process_group()
"""
from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# ── 루트 경로 설정 ────────────────────────────────────────────────────────────
_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import commons  # type: ignore[import]
import utils    # type: ignore[import]

from vits.configs.schema import VITSConfig
from vits.data.dataset import TextAudioDataset, DatasetConfig
from vits.data.collate import TextAudioCollate, TextAudioSpeakerCollate
from vits.data.sampler import DistributedBucketSampler
from vits.models.synthesizer import SynthesizerTrn
from vits.models.discriminator import MultiPeriodDiscriminator
from vits.training.losses import (
    discriminator_loss, generator_loss, feature_loss, kl_loss
)
from vits.training.mel import spec_to_mel, mel_spectrogram_torch
from vits.training.callbacks import Callback, CallbackList
from vits.training.checkpoint import CheckpointManager, CheckpointMeta

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 배치 데이터 컨테이너
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Batch:
    x:           torch.Tensor   # [B, T_text]
    x_lengths:   torch.Tensor   # [B]
    spec:        torch.Tensor   # [B, n_mels, T_spec]
    spec_lengths: torch.Tensor  # [B]
    wav:         torch.Tensor   # [B, 1, T_wav]
    wav_lengths: torch.Tensor   # [B]
    sid: Optional[torch.Tensor] = None  # [B]  (다중화자만)

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            x=self.x.to(device, non_blocking=True),
            x_lengths=self.x_lengths.to(device, non_blocking=True),
            spec=self.spec.to(device, non_blocking=True),
            spec_lengths=self.spec_lengths.to(device, non_blocking=True),
            wav=self.wav.to(device, non_blocking=True),
            wav_lengths=self.wav_lengths.to(device, non_blocking=True),
            sid=self.sid.to(device, non_blocking=True) if self.sid is not None else None,
        )


# ──────────────────────────────────────────────────────────────────────────────
# VITSTrainer
# ──────────────────────────────────────────────────────────────────────────────
class VITSTrainer:
    """
    VITS 학습기.

    Args:
        config:      VITSConfig (Pydantic) 또는 HParams 호환 객체
        rank:        현재 DDP 랭크 (단일 GPU = 0)
        world_size:  전체 DDP 프로세스 수 (단일 GPU = 1)
        device:      학습 디바이스 ('cuda', 'cpu', 'cuda:0' 등).
                     None이면 rank에서 자동 추론.
    """

    # ── 초기화 ────────────────────────────────────────────────────────────────

    def __init__(
        self,
        config: VITSConfig | Any,
        *,
        rank: int = 0,
        world_size: int = 1,
        device: Optional[str] = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        # HParams 호환 처리
        if not isinstance(config, VITSConfig):
            config = VITSConfig.from_dict(vars(config) if hasattr(config, "__dict__") else config)
        self.config = config

        self.rank       = rank
        self.world_size = world_size
        self.is_main    = rank == 0
        self.is_ddp     = world_size > 1

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")

        self.global_step = 0
        self._writer      = None
        self._writer_eval = None

        # 콜백 / 체크포인트 매니저
        self.callbacks   = CallbackList(callbacks)
        self.ckpt_mgr: CheckpointManager | None = None

        # 서브컴포넌트 (setup 메서드에서 초기화)
        self.net_g: nn.Module | None = None
        self.net_d: nn.Module | None = None
        self.optim_g: optim.Optimizer | None = None
        self.optim_d: optim.Optimizer | None = None
        self.scheduler_g = None
        self.scheduler_d = None
        self.scaler_g: GradScaler | None = None
        self.scaler_d: GradScaler | None = None
        self.train_loader: DataLoader | None = None
        self.eval_loader:  DataLoader | None = None

    # ── 공개 진입점 ───────────────────────────────────────────────────────────

    def train(self) -> None:
        """전체 학습 루프를 실행한다."""
        self._setup_logging()
        self._setup_data()
        self._setup_models()
        self._setup_optimizers()
        start_epoch = self._load_checkpoint()

        self.callbacks.on_train_begin(self)
        try:
            for epoch in range(start_epoch, self.config.train.epochs + 1):
                self.callbacks.on_epoch_begin(self, epoch)
                losses = self._train_epoch(epoch)
                self.scheduler_g.step()
                self.scheduler_d.step()
                self.callbacks.on_epoch_end(self, epoch, losses)
                if self.is_main:
                    logger.info("====> Epoch: %d | losses: %s", epoch, losses)
                # EarlyStopping 확인
                if any(getattr(cb, "stop_training", False) for cb in self.callbacks.callbacks):
                    logger.info("EarlyStopping 조건 충족 — epoch %d에서 학습 종료", epoch)
                    break
        finally:
            self.callbacks.on_train_end(self)

    # ── 설정 헬퍼 ─────────────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        if not self.is_main:
            return
        model_dir = self.config.model_dir
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        if self.config.train.log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer      = SummaryWriter(log_dir=self.config.train.log_dir)
                self._writer_eval = SummaryWriter(log_dir=os.path.join(self.config.train.log_dir, "eval"))
            except ImportError:
                logger.warning("tensorboard를 찾을 수 없습니다. 로그 기록이 비활성화됩니다.")

    def _setup_data(self) -> None:
        cfg = self.config
        ds_cfg = DatasetConfig(
            text_cleaners=cfg.data.text_cleaners,
            max_wav_value=cfg.data.max_wav_value,
            sampling_rate=cfg.data.sampling_rate,
            filter_length=cfg.data.filter_length,
            hop_length=cfg.data.hop_length,
            win_length=cfg.data.win_length,
            cleaned_text=getattr(cfg.data, "cleaned_text", False),
            add_blank=getattr(cfg.model, "add_blank", True),
        )
        train_ds = TextAudioDataset(cfg.data.training_files, ds_cfg)
        boundaries = [32, 300, 400, 500, 600, 700, 800, 900, 1000]
        train_sampler = DistributedBucketSampler(
            train_ds,
            cfg.train.batch_size,
            boundaries,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        collate_fn = (TextAudioSpeakerCollate() if train_ds.is_multispeaker else TextAudioCollate())
        self.train_loader = DataLoader(
            train_ds,
            num_workers=cfg.train.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
        )
        if self.is_main:
            eval_ds = TextAudioDataset(cfg.data.validation_files, ds_cfg)
            eval_collate = (TextAudioSpeakerCollate() if eval_ds.is_multispeaker else TextAudioCollate())
            self.eval_loader = DataLoader(
                eval_ds,
                num_workers=getattr(cfg.train, 'num_workers', 4),
                shuffle=False,
                batch_size=cfg.train.batch_size,
                pin_memory=True,
                drop_last=False,
                collate_fn=eval_collate,
            )

    def _setup_models(self) -> None:
        cfg = self.config
        n_vocab     = cfg.get_n_vocab()
        spec_ch     = cfg.data.filter_length // 2 + 1
        seg_size    = cfg.train.segment_size // cfg.data.hop_length
        model_kw    = cfg.model.model_dump()

        net_g = SynthesizerTrn(
            n_vocab, spec_ch, seg_size, **model_kw
        ).to(self.device)
        net_d = MultiPeriodDiscriminator(
            cfg.model.use_spectral_norm
        ).to(self.device)

        if self.is_ddp:
            net_g = DDP(net_g, device_ids=[self.rank])
            net_d = DDP(net_d, device_ids=[self.rank])

        self.net_g = net_g
        self.net_d = net_d

    def _setup_optimizers(self) -> None:
        cfg = self.config.train
        self.optim_g = optim.AdamW(
            self.net_g.parameters(),
            cfg.learning_rate,
            betas=tuple(cfg.betas),
            eps=cfg.eps,
        )
        self.optim_d = optim.AdamW(
            self.net_d.parameters(),
            cfg.learning_rate,
            betas=tuple(cfg.betas),
            eps=cfg.eps,
        )
        self.scaler_g = GradScaler(enabled=cfg.fp16_run)
        self.scaler_d = GradScaler(enabled=cfg.fp16_run)

    # ── 체크포인트 ────────────────────────────────────────────────────────────

    def _load_checkpoint(self) -> int:
        """최신 체크포인트를 로드하고 시작 에폭을 반환한다. 없으면 1을 반환."""
        model_dir = self.config.model_dir
        if not model_dir:
            self._setup_schedulers(start_epoch=1)
            return 1

        self.ckpt_mgr = CheckpointManager(
            save_dir=model_dir,
            keep_last_n=getattr(self.config.train, "keep_last_n_checkpoints", 3),
        )
        meta = self.ckpt_mgr.load_latest(self)
        if meta is not None:
            epoch = meta.epoch
            self.global_step = meta.step
            self._setup_schedulers(start_epoch=epoch)
            logger.info("에폭 %d (step %d) 체크포인트에서 재개합니다.", epoch, meta.step)
            return epoch
        self._setup_schedulers(start_epoch=1)
        return 1

    def _setup_schedulers(self, start_epoch: int) -> None:
        cfg = self.config.train
        self.scheduler_g = optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=cfg.lr_decay, last_epoch=start_epoch - 2
        )
        self.scheduler_d = optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=cfg.lr_decay, last_epoch=start_epoch - 2
        )

    def save_checkpoint(self, epoch: int, losses: dict[str, float] | None = None) -> None:
        """현재 모델/옵티마이저 상태를 저장한다."""
        if not self.is_main:
            return
        model_dir = self.config.model_dir
        if not model_dir:
            return
        if self.ckpt_mgr is None:
            # CheckpointManager가 아직 없으면 즉석 생성
            self.ckpt_mgr = CheckpointManager(save_dir=model_dir)
        meta = self.ckpt_mgr.save(self, epoch=epoch, losses=losses)
        self.callbacks.on_save(self, meta)

    # ── 에폭 단위 학습 ────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.train_loader.batch_sampler.set_epoch(epoch)
        self.net_g.train()
        self.net_d.train()

        epoch_losses: dict[str, float] = {}

        for batch_idx, raw_batch in enumerate(self.train_loader):
            batch = self._unpack_batch(raw_batch)
            batch = batch.to(self.device)

            self.callbacks.on_step_begin(self, self.global_step)
            step_losses = self._train_step(batch)
            self.global_step += 1

            # 에폭 평균 누적
            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            self.callbacks.on_step_end(self, self.global_step, step_losses)

            if self.is_main:
                cfg_log = self.config.train.log_interval
                if self.global_step % cfg_log == 0:
                    self._log_step(epoch, batch_idx, step_losses)

                cfg_eval = self.config.train.eval_interval
                if self.global_step % cfg_eval == 0:
                    self._evaluate(epoch)
                    self.save_checkpoint(epoch, losses=step_losses)

        n = len(self.train_loader)
        return {k: v / n for k, v in epoch_losses.items()} if n > 0 else epoch_losses

    # ── 스텝 단위 학습 ────────────────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> dict[str, float]:
        cfg = self.config
        fp16 = cfg.train.fp16_run

        # ── Generator forward ─────────────────────────────────────────────────
        with autocast(enabled=fp16):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(
                    batch.x, batch.x_lengths,
                    batch.spec, batch.spec_lengths,
                    batch.sid,
                )

            mel = spec_to_mel(
                batch.spec,
                cfg.data.filter_length,
                cfg.data.n_mel_channels,
                cfg.data.sampling_rate,
                cfg.data.mel_fmin,
                cfg.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, cfg.train.segment_size // cfg.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                cfg.data.filter_length,
                cfg.data.n_mel_channels,
                cfg.data.sampling_rate,
                cfg.data.hop_length,
                cfg.data.win_length,
                cfg.data.mel_fmin,
                cfg.data.mel_fmax,
            )
            y_wav = commons.slice_segments(
                batch.wav, ids_slice * cfg.data.hop_length, cfg.train.segment_size
            )

            # ── Discriminator update ────────────────────────────────────────
            y_d_r, y_d_g, _, _ = self.net_d(y_wav, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, _, _ = discriminator_loss(y_d_r, y_d_g)

        self.optim_d.zero_grad()
        self.scaler_d.scale(loss_disc).backward()
        self.scaler_d.unscale_(self.optim_d)
        grad_norm_d = commons.clip_grad_value_(self.net_d.parameters(), None)
        self.scaler_d.step(self.optim_d)
        self.scaler_d.update()

        # ── Generator update ───────────────────────────────────────────────
        with autocast(enabled=fp16):
            y_d_r, y_d_g, fmap_r, fmap_g = self.net_d(y_wav, y_hat)
            with autocast(enabled=False):
                loss_dur  = torch.sum(l_length.float())
                loss_mel  = F.l1_loss(y_mel, y_hat_mel) * cfg.train.c_mel
                loss_kl   = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * cfg.train.c_kl
                loss_fm   = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        self.optim_g.zero_grad()
        self.scaler_g.scale(loss_gen_all).backward()
        self.scaler_g.unscale_(self.optim_g)
        grad_norm_g = commons.clip_grad_value_(self.net_g.parameters(), None)
        self.scaler_g.step(self.optim_g)
        self.scaler_g.update()

        return {
            "loss/d/total":  loss_disc.item(),
            "loss/g/total":  loss_gen_all.item(),
            "loss/g/mel":    loss_mel.item(),
            "loss/g/dur":    loss_dur.item(),
            "loss/g/kl":     loss_kl.item(),
            "loss/g/fm":     loss_fm.item(),
            "grad_norm_g":   float(grad_norm_g) if grad_norm_g is not None else 0.0,
            "grad_norm_d":   float(grad_norm_d) if grad_norm_d is not None else 0.0,
        }

    # ── 평가 ─────────────────────────────────────────────────────────────────

    def _evaluate(self, epoch: int) -> None:
        if self.eval_loader is None:
            return
        cfg = self.config

        self.net_g.eval()
        with torch.no_grad():
            for raw_batch in self.eval_loader:
                batch = self._unpack_batch(raw_batch)
                batch = batch.to(self.device)

                # 첫 샘플만
                x       = batch.x[:1]
                x_len   = batch.x_lengths[:1]
                spec    = batch.spec[:1]
                y       = batch.wav[:1]
                y_len   = batch.wav_lengths[:1]
                sid     = batch.sid[:1] if batch.sid is not None else None

                infer_module = (
                    self.net_g.module if isinstance(self.net_g, DDP) else self.net_g
                )
                y_hat, attn, mask, _ = infer_module.infer(
                    x, x_len, sid=sid, max_len=1000
                )
                y_hat_lengths = mask.sum([1, 2]).long() * cfg.data.hop_length

                mel_gt = spec_to_mel(
                    spec, cfg.data.filter_length, cfg.data.n_mel_channels,
                    cfg.data.sampling_rate, cfg.data.mel_fmin, cfg.data.mel_fmax,
                )
                mel_gen = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    cfg.data.filter_length, cfg.data.n_mel_channels,
                    cfg.data.sampling_rate, cfg.data.hop_length, cfg.data.win_length,
                    cfg.data.mel_fmin, cfg.data.mel_fmax,
                )
                break  # 첫 배치만

        if self._writer_eval is not None:
            try:
                from utils import plot_spectrogram_to_numpy, summarize  # type: ignore[import]
                image_dict = {
                    "gen/mel": plot_spectrogram_to_numpy(mel_gen[0].cpu().numpy()),
                }
                audio_dict = {
                    "gen/audio": y_hat[0, :, : y_hat_lengths[0]],
                }
                if self.global_step == 0:
                    image_dict["gt/mel"] = plot_spectrogram_to_numpy(mel_gt[0].cpu().numpy())
                    audio_dict["gt/audio"] = y[0, :, : y_len[0]]
                summarize(
                    writer=self._writer_eval,
                    global_step=self.global_step,
                    images=image_dict,
                    audios=audio_dict,
                    audio_sampling_rate=cfg.data.sampling_rate,
                )
            except Exception as exc:
                logger.warning("평가 TensorBoard 로그 실패: %s", exc)

        self.net_g.train()

    # ── 로깅 ─────────────────────────────────────────────────────────────────

    def _log_step(self, epoch: int, batch_idx: int, losses: dict[str, float]) -> None:
        lr = self.optim_g.param_groups[0]["lr"]
        pct = 100.0 * batch_idx / max(len(self.train_loader), 1)
        logger.info(
            "Train Epoch: %d [%.0f%%] | step=%d | lr=%.2e | %s",
            epoch, pct, self.global_step, lr,
            " | ".join(f"{k}={v:.4f}" for k, v in losses.items()),
        )
        if self._writer is not None:
            try:
                from utils import summarize  # type: ignore[import]
                scalar_dict = {**losses, "learning_rate": lr}
                summarize(writer=self._writer, global_step=self.global_step, scalars=scalar_dict)
            except Exception as exc:
                logger.debug("TensorBoard 로그 실패: %s", exc)

    # ── 유틸리티 ─────────────────────────────────────────────────────────────

    def _unpack_batch(self, raw_batch) -> Batch:
        """DataLoader 출력을 Batch 컨테이너로 변환한다."""
        if len(raw_batch) == 7:
            # 다중 화자: text, text_len, spec, spec_len, wav, wav_len, sid
            x, x_len, spec, spec_len, wav, wav_len, sid = raw_batch
            return Batch(x, x_len, spec, spec_len, wav, wav_len, sid)
        else:
            # 단일 화자: text, text_len, spec, spec_len, wav, wav_len
            x, x_len, spec, spec_len, wav, wav_len = raw_batch[:6]
            return Batch(x, x_len, spec, spec_len, wav, wav_len, None)
