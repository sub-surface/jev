import numpy as np
import pytest
import torch

from calib import escalation, metrics, recalibrate, scoring


@pytest.mark.parametrize("rule", ["log", "brier", "spherical"])
def test_rules_are_proper(rule):
    """Expected loss under q is minimised by reporting q (numerically, vs many perturbations)."""
    torch.manual_seed(0)
    q = torch.softmax(torch.randn(5), 0)
    ys = torch.arange(5)

    def expected(logits):
        L = scoring.proper_loss(logits.expand(5, -1), ys, rule=rule)
        return (L * q).sum()

    best = expected(q.log())
    for _ in range(200):
        assert expected(q.log() + 0.3 * torch.randn(5)) >= best - 1e-6


def test_rps_is_proper_and_distance_aware():
    q = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1])
    ys = torch.arange(5)
    exp = lambda p: (scoring.rps_loss(p.expand(5, -1), ys) * q).sum()
    assert exp(q) < exp(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]))
    # mass one step off is penalised less than mass far off
    y = torch.tensor([2])
    near = scoring.rps_loss(torch.tensor([[0, 0, 0, 1.0, 0]]), y)
    far = scoring.rps_loss(torch.tensor([[0, 0, 0, 0, 1.0]]), y)
    assert near < far


def test_mask_handles_mixed_cardinality():
    logits = torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.5, 0.1, 0.3, 2.0]])
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    y = torch.tensor([1, 3])
    a = scoring.proper_loss(logits, y, mask, rule="log")
    assert torch.isclose(a[0], torch.nn.functional.cross_entropy(logits[:1, :2], y[:1]))
    b = scoring.proper_loss(logits, y, mask, rule="brier")
    assert torch.isfinite(b).all()


def _calibrated_sample(n, rng):
    conf = rng.uniform(0.3, 1.0, n)
    return conf, (rng.uniform(size=n) < conf).astype(float)


def test_calibration_metrics_near_zero_when_calibrated():
    rng = np.random.default_rng(0)
    conf, out = _calibrated_sample(20000, rng)
    assert metrics.ece(conf, out) < 0.02
    assert metrics.ece_l2_debiased(conf, out) < 0.02
    assert metrics.smooth_ece(conf, out) < 0.03


def test_calibration_metrics_detect_overconfidence():
    rng = np.random.default_rng(1)
    conf, out = _calibrated_sample(20000, rng)
    over = np.clip(conf + 0.15, 0, 1)
    assert metrics.ece(over, out) > 0.1
    assert metrics.smooth_ece(over, out) > 0.08


def test_debiasing_helps_at_small_n():
    rng = np.random.default_rng(2)
    plug, deb = [], []
    for _ in range(200):
        conf, out = _calibrated_sample(300, rng)
        plug.append(metrics.ece(conf, out))
        deb.append(metrics.ece_l2_debiased(conf, out))
    assert np.mean(deb) < np.mean(plug)


def test_temperature_scaling_recovers_temperature():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(20000, 4)) * 2
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=pi) for pi in p])
    ts = recalibrate.TemperatureScaler().fit(z * 3.0, y)  # logits 3x too sharp
    assert abs(ts.T - 3.0) < 0.15


def test_isotonic_is_monotone():
    rng = np.random.default_rng(4)
    conf, out = _calibrated_sample(2000, rng)
    iso = recalibrate.IsotonicConfidence().fit(conf, out)
    g = iso(np.linspace(0, 1, 50))
    assert np.all(np.diff(g) >= -1e-12)


def test_escalation_curve_endpoints_and_ordering():
    rng = np.random.default_rng(5)
    conf, s1 = _calibrated_sample(5000, rng)
    s2 = np.ones_like(s1)
    c = escalation.escalation_curve(conf, s1, s2)
    assert np.isclose(c["gated"][0], s1.mean()) and np.isclose(c["gated"][-1], 1.0)
    assert np.all(c["oracle"] >= c["gated"] - 1e-9)
    assert c["gated"][30] > c["random"][30]
    assert escalation.escalation_needed(c) < escalation.escalation_needed(c, key="random")


@pytest.mark.parametrize("score", ["log", "brier"])
def test_corp_identity_and_signs(score):
    rng = np.random.default_rng(6)
    x = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < x).astype(float)
    for p in [x, np.clip(x + 0.2, 0.01, 0.99), np.full(5000, 0.5), np.round(x, 1)]:
        d = metrics.corp_decomposition(p, y, score)
        assert abs(d["score"] - (d["MCB"] - d["DSC"] + d["UNC"])) < 1e-9
        assert d["MCB"] >= -1e-12 and d["DSC"] >= -1e-12
    # a calibrated sharp forecaster has small MCB and large DSC; a constant one has DSC = 0
    good = metrics.corp_decomposition(x, y, score)
    flat = metrics.corp_decomposition(np.full(5000, y.mean()), y, score)
    assert good["MCB"] < 0.02 and good["DSC"] > 0.05 and abs(flat["DSC"]) < 1e-12
