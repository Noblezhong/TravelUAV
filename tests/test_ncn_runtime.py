from src.vlnce_src.ncn_runtime import EdgeLatencyEstimate


def test_edge_latency_estimate_uses_warmup_then_ema():
    estimate = EdgeLatencyEstimate(compute_ms=100.0, alpha=0.5)
    assert estimate.remaining_ms(0.0, 25.0, 20.0) == 95.0
    estimate.update(200.0, 100.0)
    assert estimate.compute_ms == 200.0


def test_edge_latency_estimate_has_no_negative_remaining_time():
    estimate = EdgeLatencyEstimate(compute_ms=10.0)
    assert estimate.remaining_ms(0.0, 100.0, 10.0) == 0.0
