"""
손실 함수 단위 테스트.
"""
import pytest
import sys
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


B, C, T = 2, 64, 100  # 배치, 채널, 시간


class TestAdversarialLosses:
    """GAN 손실 함수 테스트."""

    def test_discriminator_loss_shape(self):
        """discriminator_loss가 스칼라를 반환하는지 확인."""
        from losses import discriminator_loss

        # 판별자 출력 시뮬레이션 (각 period/scale 판별자)
        real_outputs = [torch.rand(B, T) for _ in range(5)]
        fake_outputs = [torch.rand(B, T) for _ in range(5)]

        loss, r_losses, g_losses = discriminator_loss(real_outputs, fake_outputs)

        assert loss.ndim == 0, "총 판별자 loss는 스칼라여야 함"
        assert len(r_losses) == 5
        assert len(g_losses) == 5

    def test_discriminator_loss_positive(self):
        """판별자 loss는 항상 0 이상이어야 함 (LSGAN)."""
        from losses import discriminator_loss

        real_outputs = [torch.rand(B, T) for _ in range(3)]
        fake_outputs = [torch.rand(B, T) for _ in range(3)]
        loss, _, _ = discriminator_loss(real_outputs, fake_outputs)

        assert loss.item() >= 0.0

    def test_generator_loss_shape(self):
        """generator_loss가 스칼라를 반환하는지 확인."""
        from losses import generator_loss

        disc_outputs = [torch.rand(B, T) for _ in range(5)]
        loss, gen_losses = generator_loss(disc_outputs)

        assert loss.ndim == 0
        assert len(gen_losses) == 5

    def test_generator_loss_positive(self):
        """생성자 loss는 항상 0 이상이어야 함."""
        from losses import generator_loss

        disc_outputs = [torch.rand(B, T) for _ in range(3)]
        loss, _ = generator_loss(disc_outputs)

        assert loss.item() >= 0.0


class TestFeatureLoss:
    """Feature matching loss 테스트."""

    def test_feature_loss_shape(self):
        """feature_loss가 스칼라를 반환하는지 확인."""
        from losses import feature_loss

        # fmap: 각 판별자에서 여러 레이어의 특성 맵
        fmap_real = [[torch.randn(B, 16, T) for _ in range(4)] for _ in range(5)]
        fmap_fake = [[torch.randn(B, 16, T) for _ in range(4)] for _ in range(5)]

        loss = feature_loss(fmap_real, fmap_fake)
        assert loss.ndim == 0

    def test_feature_loss_zero_when_same(self):
        """동일한 특성 맵이면 loss가 0이어야 함."""
        from losses import feature_loss

        fmap = [[torch.randn(B, 16, T) for _ in range(4)] for _ in range(5)]
        loss = feature_loss(fmap, fmap)

        assert abs(loss.item()) < 1e-5, f"동일 입력 feature loss != 0: {loss.item()}"

    def test_feature_loss_positive(self):
        """feature matching loss는 항상 0 이상."""
        from losses import feature_loss

        fmap_r = [[torch.randn(B, 16, T) for _ in range(4)] for _ in range(5)]
        fmap_g = [[torch.randn(B, 16, T) for _ in range(4)] for _ in range(5)]

        loss = feature_loss(fmap_r, fmap_g)
        assert loss.item() >= 0.0


class TestKLLoss:
    """KL divergence loss 테스트."""

    def test_kl_loss_shape(self):
        """kl_loss가 스칼라를 반환하는지 확인."""
        from losses import kl_loss

        z_p = torch.randn(B, C, T)
        logs_q = torch.zeros(B, C, T)
        m_p = torch.zeros(B, C, T)
        logs_p = torch.zeros(B, C, T)
        z_mask = torch.ones(B, 1, T)

        loss = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        assert loss.ndim == 0

    def test_kl_loss_near_zero_for_matched_distributions(self):
        """z_p ~ N(m_p, exp(logs_p))이고 prior == posterior이면 KL이 0에 가까워야 함.
        
        구현된 공식: kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p)^2 * exp(-2*logs_p)
        z_p = m_p일 때: kl = logs_p - logs_q - 0.5
        logs_p == logs_q == 0일 때: kl = -0.5 (per element)
        E[kl] with z_p ~ N(m_p, exp(logs_p)): E = -0.5 + 0.5 = 0
        샘플 평균은 0 근처여야 함 (±0.1 허용).
        """
        from losses import kl_loss

        torch.manual_seed(42)
        # 많은 샘플에서 평균이 0에 수렴해야 함
        z_p = torch.randn(1, 64, 10000)  # 대규모 샘플
        logs_q = torch.zeros(1, 64, 10000)
        m_p = torch.zeros(1, 64, 10000)
        logs_p = torch.zeros(1, 64, 10000)
        z_mask = torch.ones(1, 1, 10000)

        loss = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        # 평균 KL은 0에 수렴 (허용 오차 ±0.1)
        assert abs(loss.item()) < 0.1, f"기대값이 0에 가까워야 함, got {loss.item():.4f}"

    def test_kl_loss_with_mask(self):
        """마스크가 일부만 1일 때 마스킹이 적용되는지 확인."""
        from losses import kl_loss

        z_p = torch.randn(B, C, T)
        logs_q = torch.zeros(B, C, T)
        m_p = torch.zeros(B, C, T)
        logs_p = torch.zeros(B, C, T)

        # 전체 마스크
        mask_full = torch.ones(B, 1, T)
        # 절반 마스크
        mask_half = torch.zeros(B, 1, T)
        mask_half[:, :, :T//2] = 1.0

        loss_full = kl_loss(z_p, logs_q, m_p, logs_p, mask_full)
        loss_half = kl_loss(z_p, logs_q, m_p, logs_p, mask_half)

        # 마스크 절반이면 loss가 더 작아야 함
        # (정확히 절반은 아니지만 분명히 차이가 있어야 함)
        assert loss_full.item() != loss_half.item()


class TestMelProcessing:
    """mel_processing.py 유틸 함수 테스트."""

    def test_spectrogram_torch_shape(self):
        """spectrogram_torch 출력 shape 검증."""
        from mel_processing import spectrogram_torch

        audio = torch.randn(2, 1, 22050)  # 1초 배치
        audio = audio.squeeze(1)  # [B, T]

        spec = spectrogram_torch(
            audio, n_fft=1024, sampling_rate=22050,
            hop_size=256, win_size=1024
        )
        assert spec.ndim == 3
        assert spec.shape[1] == 513  # n_fft // 2 + 1

    def test_mel_spectrogram_torch_shape(self):
        """mel_spectrogram_torch 출력 shape 검증."""
        from mel_processing import mel_spectrogram_torch

        audio = torch.randn(2, 22050)  # [B, T]

        mel = mel_spectrogram_torch(
            audio, n_fft=1024, num_mels=80,
            sampling_rate=22050, hop_size=256,
            win_size=1024, fmin=0.0, fmax=None
        )
        assert mel.ndim == 3
        assert mel.shape[1] == 80   # n_mel_channels

    def test_spec_to_mel_torch_shape(self):
        """spec_to_mel_torch 출력 shape 검증."""
        from mel_processing import spec_to_mel_torch

        spec = torch.randn(2, 513, 100)  # [B, n_fft//2+1, T]

        mel = spec_to_mel_torch(
            spec, n_fft=1024, num_mels=80,
            sampling_rate=22050, fmin=0.0, fmax=None
        )
        assert mel.shape == (2, 80, 100)
