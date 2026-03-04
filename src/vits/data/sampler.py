"""
DistributedBucketSampler.

비슷한 스펙트로그램 길이의 샘플들을 같은 배치에 묶어
패딩 낭비를 줄인다.

단일 GPU 학습에서도 사용할 수 있도록 num_replicas=1, rank=0 을 기본으로 지원한다.
"""
from __future__ import annotations

import torch
import torch.utils.data


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    길이 경계(boundaries)로 정의된 버킷에 샘플을 할당하고,
    DDP 환경에서 각 랭크에 균일하게 분배한다.

    Args:
        dataset:      ``lengths`` 속성을 가진 Dataset
        batch_size:   배치 크기 (랭크당)
        boundaries:   길이 경계 리스트. 예) [0, 100, 200, 300]
                      → 버킷 (0,100], (100,200], (200,300]
        num_replicas: DDP 프로세스 수 (None = 환경변수 WORLD_SIZE에서 감지)
        rank:         현재 랭크 (None = 환경변수 RANK에서 감지)
        shuffle:      에폭마다 순서를 섞는지 여부

    Notes:
        - boundaries 범위 밖 샘플은 무시된다.
        - 버킷이 비어있으면 해당 버킷은 제거된다.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        boundaries: list[int],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.lengths: list[int] = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = list(boundaries)  # 수정 가능하도록 복사

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    # ── 버킷 생성 ─────────────────────────────────────────────────────────────

    def _create_buckets(self) -> tuple[list[list[int]], list[int]]:
        """샘플을 버킷에 할당하고 DDP 균등 분할을 위한 패딩 크기를 계산한다."""
        n_buckets = len(self.boundaries) - 1
        buckets: list[list[int]] = [[] for _ in range(n_buckets)]

        for idx, length in enumerate(self.lengths):
            b = self._bisect(length)
            if b != -1:
                buckets[b].append(idx)

        # 빈 버킷 제거
        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        # DDP 균등 분할을 위한 패딩 계산
        total_batch_size = self.num_replicas * self.batch_size
        num_samples_per_bucket: list[int] = []
        for bucket in buckets:
            rem = (total_batch_size - (len(bucket) % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len(bucket) + rem)

        return buckets, num_samples_per_bucket

    # ── 이진 탐색 ─────────────────────────────────────────────────────────────

    def _bisect(self, x: int, lo: int = 0, hi: int | None = None) -> int:
        """x가 속하는 버킷 인덱스를 반환한다. 범위 밖이면 -1."""
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        return -1

    # ── 이터레이터 ────────────────────────────────────────────────────────────

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # 버킷별 인덱스 순열
        if self.shuffle:
            bucket_indices = [
                torch.randperm(len(b), generator=g).tolist()
                for b in self.buckets
            ]
        else:
            bucket_indices = [list(range(len(b))) for b in self.buckets]

        batches: list[list[int]] = []
        for i, bucket in enumerate(self.buckets):
            ids = bucket_indices[i]
            len_bucket = len(bucket)
            n = self.num_samples_per_bucket[i]

            # 부족분 채우기
            rem = n - len_bucket
            ids = ids + ids * (rem // len_bucket) + ids[: (rem % len_bucket)]

            # 랭크별 슬라이스
            ids = ids[self.rank :: self.num_replicas]

            # 배치 구성
            for j in range(len(ids) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids[j * self.batch_size : (j + 1) * self.batch_size]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[k] for k in batch_order]

        self.batches = batches
        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
