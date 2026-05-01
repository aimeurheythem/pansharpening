"""
test_metrics.py — Unit tests for the metrics module.

Run: pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import sam, ergas, q4, scc, psnr, ssim_metric, compute_all_metrics, MetricTracker


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def perfect_pair():
    """GT and prediction are identical — all metrics should be ideal."""
    rng = np.random.default_rng(42)
    img = rng.random((8, 64, 64), dtype=np.float32)
    return img, img.copy()


@pytest.fixture
def noisy_pair():
    """GT and prediction with small Gaussian noise."""
    rng = np.random.default_rng(42)
    gt   = rng.random((8, 64, 64), dtype=np.float32)
    pred = np.clip(gt + rng.normal(0, 0.02, gt.shape), 0, 1).astype(np.float32)
    return gt, pred


@pytest.fixture
def bad_pair():
    """GT and completely different random image."""
    rng = np.random.default_rng(0)
    gt   = rng.random((8, 64, 64), dtype=np.float32)
    pred = rng.random((8, 64, 64), dtype=np.float32)
    return gt, pred


# =============================================================================
# PERFECT CASE
# =============================================================================

class TestPerfectCase:
    def test_sam_perfect(self, perfect_pair):
        gt, pred = perfect_pair
        assert sam(gt, pred) == pytest.approx(0.0, abs=1e-2)

    def test_ergas_perfect(self, perfect_pair):
        gt, pred = perfect_pair
        assert ergas(gt, pred) == pytest.approx(0.0, abs=1e-4)

    def test_q4_perfect(self, perfect_pair):
        gt, pred = perfect_pair
        score = q4(gt, pred)
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_psnr_perfect(self, perfect_pair):
        gt, pred = perfect_pair
        score = psnr(gt, pred)
        assert score == float("inf") or score > 100.0

    def test_ssim_perfect(self, perfect_pair):
        gt, pred = perfect_pair
        score = ssim_metric(gt, pred)
        assert score == pytest.approx(1.0, abs=1e-3)


# =============================================================================
# ORDERING (better quality → better metrics)
# =============================================================================

class TestOrdering:
    def test_sam_ordering(self, perfect_pair, noisy_pair, bad_pair):
        sam_perfect = sam(*perfect_pair)
        sam_noisy   = sam(*noisy_pair)
        sam_bad     = sam(*bad_pair)
        assert sam_perfect <= sam_noisy <= sam_bad

    def test_ergas_ordering(self, perfect_pair, noisy_pair, bad_pair):
        e_perfect = ergas(*perfect_pair)
        e_noisy   = ergas(*noisy_pair)
        e_bad     = ergas(*bad_pair)
        assert e_perfect <= e_noisy <= e_bad

    def test_psnr_ordering(self, perfect_pair, noisy_pair, bad_pair):
        p_perfect = psnr(*perfect_pair)
        p_noisy   = psnr(*noisy_pair)
        p_bad     = psnr(*bad_pair)
        assert p_perfect >= p_noisy >= p_bad

    def test_ssim_ordering(self, perfect_pair, noisy_pair, bad_pair):
        s_perfect = ssim_metric(*perfect_pair)
        s_noisy   = ssim_metric(*noisy_pair)
        s_bad     = ssim_metric(*bad_pair)
        assert s_perfect >= s_noisy >= s_bad


# =============================================================================
# VALUE RANGES
# =============================================================================

class TestRanges:
    def test_sam_range(self, bad_pair):
        """SAM should be in [0, 180] degrees."""
        score = sam(*bad_pair)
        assert 0.0 <= score <= 180.0

    def test_q4_range(self, bad_pair):
        """Q4 should be in [-1, 1]."""
        score = q4(*bad_pair)
        assert -1.0 <= score <= 1.0

    def test_scc_range(self, bad_pair):
        """SCC should be in [-1, 1]."""
        score = scc(*bad_pair)
        assert -1.0 <= score <= 1.0

    def test_psnr_positive(self, noisy_pair):
        """PSNR for non-identical images should be positive and finite."""
        score = psnr(*noisy_pair)
        assert np.isfinite(score) and score > 0

    def test_ssim_range(self, bad_pair):
        """SSIM should be in [0, 1] for non-negative images."""
        score = ssim_metric(*bad_pair)
        assert 0.0 <= score <= 1.0


# =============================================================================
# COMPUTE ALL METRICS
# =============================================================================

class TestComputeAll:
    def test_returns_all_keys(self, noisy_pair):
        metrics = compute_all_metrics(*noisy_pair)
        expected = {"SAM", "ERGAS", "Q4", "SCC", "PSNR", "SSIM"}
        assert set(metrics.keys()) == expected

    def test_all_finite(self, noisy_pair):
        metrics = compute_all_metrics(*noisy_pair)
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_shape_mismatch_raises(self):
        gt   = np.random.rand(4, 64, 64).astype(np.float32)
        pred = np.random.rand(4, 32, 32).astype(np.float32)
        with pytest.raises(AssertionError):
            compute_all_metrics(gt, pred)


# =============================================================================
# METRIC TRACKER
# =============================================================================

class TestMetricTracker:
    def test_update_and_compute(self, noisy_pair):
        tracker = MetricTracker()
        gt, pred = noisy_pair
        tracker.update(gt, pred)
        results = tracker.compute()
        assert len(results) == 6

    def test_reset(self, noisy_pair):
        tracker = MetricTracker()
        tracker.update(*noisy_pair)
        tracker.reset()
        # After reset, compute should return empty (no values)
        results = tracker.compute()
        assert results == {}

    def test_batch_update(self):
        rng  = np.random.default_rng(0)
        gt   = rng.random((4, 8, 64, 64)).astype(np.float32)
        pred = np.clip(gt + rng.normal(0, 0.02, gt.shape), 0, 1).astype(np.float32)
        tracker = MetricTracker()
        tracker.update_batch(gt, pred)
        results = tracker.compute()
        assert len(results) == 6
        assert results["SAM"] >= 0.0
