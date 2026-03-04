"""
기존 모델 클래스의 forward pass shape 테스트.
리팩토링 후에도 동일하게 통과해야 한다.
"""
import pytest
import sys
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─── 테스트용 소형 모델 파라미터 ──────────────────────────────────────────────
SMALL_MODEL_KWARGS = dict(
    n_vocab=66,
    spec_channels=513,
    segment_size=32,         # 8192 // 256 = 32
    inter_channels=64,
    hidden_channels=64,
    filter_channels=128,
    n_heads=2,
    n_layers=2,
    kernel_size=3,
    p_dropout=0.0,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[8, 8, 2, 2],
    upsample_initial_channel=128,
    upsample_kernel_sizes=[16, 16, 4, 4],
    n_speakers=0,
    gin_channels=0,
    use_sdp=True,
)

BATCH = 2
T_TEXT = 30
T_SPEC = 150
T_AUDIO = 38400  # T_SPEC * hop_length(256)


@pytest.fixture(scope="module")
def single_speaker_model():
    """싱글 화자 SynthesizerTrn 인스턴스."""
    from models import SynthesizerTrn
    model = SynthesizerTrn(**SMALL_MODEL_KWARGS)
    model.eval()
    return model


@pytest.fixture(scope="module")
def multi_speaker_model():
    """멀티 화자 SynthesizerTrn 인스턴스."""
    from models import SynthesizerTrn
    kwargs = dict(SMALL_MODEL_KWARGS, n_speakers=10, gin_channels=64)
    model = SynthesizerTrn(**kwargs)
    model.eval()
    return model


class TestSynthesizerTrnShapes:
    """SynthesizerTrn forward/infer 출력 shape 검증."""

    def test_forward_output_types(self, single_speaker_model):
        """forward()가 튜플을 반환하는지 확인.
        
        SynthesizerTrn.forward(x, x_lengths, y, y_lengths, sid=None)
        여기서 y = spec (spectrogram), y_lengths = spec_lengths
        """
        x = torch.randint(0, 66, (BATCH, T_TEXT))
        x_lengths = torch.LongTensor([T_TEXT, 20])
        spec = torch.randn(BATCH, 513, T_SPEC)
        spec_lengths = torch.LongTensor([T_SPEC, 100])

        with torch.no_grad():
            result = single_speaker_model(x, x_lengths, spec, spec_lengths)

        assert isinstance(result, tuple)
        assert len(result) == 7

    def test_forward_audio_output_shape(self, single_speaker_model):
        """출력 오디오 shape이 [B, 1, T] 형태인지 확인."""
        x = torch.randint(0, 66, (BATCH, T_TEXT))
        x_lengths = torch.LongTensor([T_TEXT, 20])
        spec = torch.randn(BATCH, 513, T_SPEC)
        spec_lengths = torch.LongTensor([T_SPEC, 100])

        with torch.no_grad():
            o, l_length, attn, ids_slice, x_mask, z_mask, extras = single_speaker_model(
                x, x_lengths, spec, spec_lengths
            )

        assert o.ndim == 3, f"오디오 출력은 3차원이어야 함, got {o.ndim}"
        assert o.shape[0] == BATCH, f"배치 크기 불일치: {o.shape[0]} != {BATCH}"
        assert o.shape[1] == 1, f"채널 수는 1(모노)이어야 함: {o.shape[1]}"

    def test_forward_attention_shape(self, single_speaker_model):
        """attention shape이 [B, 1, T_text, T_text] 형태인지 확인."""
        x = torch.randint(0, 66, (BATCH, T_TEXT))
        x_lengths = torch.LongTensor([T_TEXT, 20])
        spec = torch.randn(BATCH, 513, T_SPEC)
        spec_lengths = torch.LongTensor([T_SPEC, 100])

        with torch.no_grad():
            o, l_length, attn, ids_slice, x_mask, z_mask, extras = single_speaker_model(
                x, x_lengths, spec, spec_lengths
            )

        assert attn.ndim == 4, f"attention은 4차원이어야 함: {attn.shape}"
        assert attn.shape[0] == BATCH

    def test_forward_duration_loss_scalar(self, single_speaker_model):
        """duration loss가 스칼라인지 확인."""
        x = torch.randint(0, 66, (BATCH, T_TEXT))
        x_lengths = torch.LongTensor([T_TEXT, 20])
        spec = torch.randn(BATCH, 513, T_SPEC)
        spec_lengths = torch.LongTensor([T_SPEC, 100])

        with torch.no_grad():
            o, l_length, *_ = single_speaker_model(
                x, x_lengths, spec, spec_lengths
            )

        assert l_length.numel() >= 1, "duration loss는 최소 1개 이상의 요소를 가져야 함"

    def test_infer_output_shape(self, single_speaker_model):
        """infer()가 오디오를 생성하는지 확인."""
        x = torch.randint(0, 66, (1, T_TEXT))
        x_lengths = torch.LongTensor([T_TEXT])

        with torch.no_grad():
            o, attn, mask, *_ = single_speaker_model.infer(x, x_lengths, max_len=100)

        assert o.ndim == 3
        assert o.shape[0] == 1
        assert o.shape[1] == 1

    def test_multi_speaker_forward(self, multi_speaker_model):
        """멀티 화자 forward가 동작하는지 확인."""
        x = torch.randint(0, 66, (BATCH, T_TEXT))
        x_lengths = torch.LongTensor([T_TEXT, 20])
        spec = torch.randn(BATCH, 513, T_SPEC)
        spec_lengths = torch.LongTensor([T_SPEC, 100])
        sid = torch.LongTensor([0, 1])

        with torch.no_grad():
            result = multi_speaker_model(
                x, x_lengths, spec, spec_lengths, sid=sid
            )

        o = result[0]
        assert o.shape[0] == BATCH

    def test_voice_conversion(self, multi_speaker_model):
        """voice_conversion()이 동작하는지 확인."""
        spec = torch.randn(1, 513, T_SPEC)
        spec_lengths = torch.LongTensor([T_SPEC])
        sid_src = torch.LongTensor([0])
        sid_tgt = torch.LongTensor([1])

        with torch.no_grad():
            o_hat, mask, *_ = multi_speaker_model.voice_conversion(
                spec, spec_lengths, sid_src, sid_tgt
            )

        assert o_hat.ndim == 3
        assert o_hat.shape[0] == 1


class TestSubModuleShapes:
    """핵심 서브모듈 shape 테스트."""

    def test_text_encoder_output_shape(self):
        """TextEncoder 출력 shape 검증."""
        from models import TextEncoder
        enc = TextEncoder(
            n_vocab=66, out_channels=64, hidden_channels=64,
            filter_channels=128, n_heads=2, n_layers=2,
            kernel_size=3, p_dropout=0.0
        )
        enc.eval()
        x = torch.randint(0, 66, (BATCH, T_TEXT))
        x_lengths = torch.LongTensor([T_TEXT, 20])

        with torch.no_grad():
            x_out, m, logs, x_mask = enc(x, x_lengths)

        assert x_out.shape == (BATCH, 64, T_TEXT)
        assert m.shape == (BATCH, 64, T_TEXT)
        assert logs.shape == (BATCH, 64, T_TEXT)
        assert x_mask.shape == (BATCH, 1, T_TEXT)

    def test_posterior_encoder_output_shape(self):
        """PosteriorEncoder 출력 shape 검증."""
        from models import PosteriorEncoder
        enc = PosteriorEncoder(
            in_channels=513, out_channels=64, hidden_channels=64,
            kernel_size=5, dilation_rate=1, n_layers=4
        )
        enc.eval()
        spec = torch.randn(BATCH, 513, T_SPEC)
        spec_lengths = torch.LongTensor([T_SPEC, 100])

        with torch.no_grad():
            z, m, logs, mask = enc(spec, spec_lengths)

        assert z.shape == (BATCH, 64, T_SPEC)
        assert m.shape == (BATCH, 64, T_SPEC)

    def test_generator_output_shape(self):
        """Generator(HiFi-GAN decoder) 출력 shape 검증."""
        from models import Generator
        gen = Generator(
            initial_channel=64,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=128,
            upsample_kernel_sizes=[16, 16, 4, 4],
        )
        gen.eval()
        z = torch.randn(BATCH, 64, 32)  # segment_size // 256

        with torch.no_grad():
            audio = gen(z)

        assert audio.ndim == 3
        assert audio.shape[0] == BATCH
        assert audio.shape[1] == 1
        assert audio.shape[2] == 32 * 256  # upsample 256배

    def test_discriminator_output_types(self):
        """MultiPeriodDiscriminator 출력 형태 검증."""
        from models import MultiPeriodDiscriminator
        disc = MultiPeriodDiscriminator(use_spectral_norm=False)
        disc.eval()

        audio_real = torch.randn(BATCH, 1, 8192)
        audio_fake = torch.randn(BATCH, 1, 8192)

        with torch.no_grad():
            y_r, y_g, fmap_r, fmap_g = disc(audio_real, audio_fake)

        assert isinstance(y_r, list)
        assert isinstance(y_g, list)
        assert isinstance(fmap_r, list)
        assert isinstance(fmap_g, list)
        assert len(y_r) == len(y_g)


class TestMonotonicAlign:
    """monotonic_align 모듈 테스트."""

    def test_maximum_path_output_shape(self):
        """maximum_path가 올바른 shape을 반환하는지."""
        try:
            import monotonic_align
        except ImportError:
            pytest.skip("monotonic_align 빌드 안 됨 (make build-ext 실행 필요)")

        B, T_t, T_s = 2, 50, 30
        neg_cent = torch.randn(B, T_t, T_s)
        mask = torch.ones(B, T_t, T_s)

        path = monotonic_align.maximum_path(neg_cent, mask)
        assert path.shape == (B, T_t, T_s)

    def test_maximum_path_valid_path(self):
        """monotonic path가 각 행에서 0 또는 1 값만 가지는지."""
        try:
            import monotonic_align
        except ImportError:
            pytest.skip("monotonic_align 빌드 안 됨")

        B, T_t, T_s = 1, 20, 10
        neg_cent = torch.randn(B, T_t, T_s)
        mask = torch.ones(B, T_t, T_s)

        path = monotonic_align.maximum_path(neg_cent, mask)
        unique_vals = path.unique().tolist()
        for v in unique_vals:
            assert v in [0.0, 1.0], f"path에 0/1 외 값 존재: {v}"
