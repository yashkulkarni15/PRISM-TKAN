"""Tests for the white-box *physics* XAI module (delay / regenerative checks).

Regenerative milling chatter is a delay differential equation: the model carries
a kinematic delay ``tau = round(fs / tpf)`` (the tooth-pass period in samples).
These tests probe whether the *delay* is load-bearing, using a small
differentiable **regenerative stub** as ground truth:

    logit = sum_t w[t, 0] * w[t - tau, 0]

so a signal that is periodic with period ``TRUE_TAU`` on channel 0 lines up with
its own delayed copy (large logit), while independent noise does not (logit ~ 0).
The delay-ablation, tau-sweep, weight-share, and attribution-spectrum probes are
all checked against this analytically-known structure.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from pri2m.models.tkan import delayed_input  # noqa: E402
from pri2m.models.tkan_model import TKANModel  # noqa: E402
from pri2m.xai import physics  # noqa: E402
from pri2m.xai.activations import collect_activations  # noqa: E402
from pri2m.xai.semantics import edge_index_map  # noqa: E402

_FS = 10_000.0
_L = 40
_TRUE_TAU = 8
_TPF = _FS / _TRUE_TAU  # so round(fs / tpf) == TRUE_TAU
_BIG_TAU = 10**6

# Two distinct operating points for the per-window-relative sweep: windows whose
# own tau differs (6 vs 9 samples), so one *global* delay cannot suit both.
_TAU_A = 6
_TAU_B = 9


# --------------------------------------------------------------------------- #
# Differentiable regenerative stubs (ground truth)
# --------------------------------------------------------------------------- #
class Regen(nn.Module):
    """A differentiable regenerative DDE stub.

    ``forward(batch)`` reads ``window`` ``[B, L, C]`` and ``tau`` (int or
    ``LongTensor[B]``), builds the delayed sequence ``wd[t] = w[t - tau]`` via
    :func:`delayed_input`, and returns the self-product logit
    ``sum_t w[:, t, 0] * wd[:, t, 0]`` as both heads (shape ``[B]``).
    """

    def forward(self, batch: dict) -> dict:
        w = batch["window"]
        tau = batch["tau"]
        if isinstance(tau, torch.Tensor):
            tau = int(tau[0])
        B, L, C = w.shape
        wd = torch.stack([delayed_input(w, t, tau) for t in range(L)], dim=1)  # [B, L, C]
        logit = (w[:, :, 0] * wd[:, :, 0]).sum(dim=1)  # [B]
        return {"chatter_logit": logit, "b_over_blim": logit}


class PerWindowRegen(nn.Module):
    """Regenerative DDE stub that honours a *per-window* ``tau``.

    Unlike :class:`Regen` (which collapses ``tau`` to its first element), this
    stub uses the full ``LongTensor[B]`` delay so :func:`delayed_input` applies a
    different delay to each window. That is exactly what the per-window-relative
    tau-sweep exercises: each window is delayed by a multiple of *its own* tau.
    """

    def forward(self, batch: dict) -> dict:
        w = batch["window"]
        tau = batch["tau"]
        if not isinstance(tau, torch.Tensor):
            tau = torch.full((w.shape[0],), int(tau), dtype=torch.long)
        _, L, _ = w.shape
        wd = torch.stack([delayed_input(w, t, tau) for t in range(L)], dim=1)  # [B, L, C]
        logit = (w[:, :, 0] * wd[:, :, 0]).sum(dim=1)  # [B]
        return {"chatter_logit": logit, "b_over_blim": logit}


class FreqStub(nn.Module):
    """A stub whose logit correlates the input with a fixed reference envelope.

    ``chatter_logit = sum_t w[:, t, 0] * ref[t]`` with a *strictly positive*
    F0-modulated envelope ``ref[t] = 1 + 0.9 cos(2 pi F0 t / fs)``. Pairing that
    with a channel-0 window equal to ``ref`` makes the Integrated-Gradients
    attribution onto channel 0 equal to ``ref[t]**2`` -- non-negative, so the
    magnitude ``|IG|`` does not rectify-double a zero-mean tone, and the
    attribution power spectrum's fundamental lands at ``F0``.

    ``ref`` is exposed via :meth:`reference` so the test can build the matching
    window without re-deriving the envelope.
    """

    def __init__(self, f0: float, fs: float, length: int) -> None:
        super().__init__()
        t = torch.arange(length, dtype=torch.float32)
        self.register_buffer("ref", 1.0 + 0.9 * torch.cos(2 * np.pi * f0 * t / fs))  # [L]

    def reference(self) -> np.ndarray:
        """The strictly-positive F0 envelope ``ref`` as ``np[L]``."""
        return self.ref.detach().cpu().numpy()

    def forward(self, batch: dict) -> dict:
        w = batch["window"]
        logit = (w[:, :, 0] * self.ref).sum(dim=1)  # [B]
        return {"chatter_logit": logit, "b_over_blim": logit}


# --------------------------------------------------------------------------- #
# Window construction: periodic positives, noisy negatives
# --------------------------------------------------------------------------- #
def _regen_batch(n_pos: int = 24, n_neg: int = 24, seed: int = 0):
    """A batch whose positives are period-TRUE_TAU tones, negatives are noise.

    Returns ``(batch, labels)`` with ``batch["window"]`` ``[B, L, C=4]``,
    ``batch["tau"]`` a ``LongTensor[B]`` of ``TRUE_TAU``, and float labels
    (1 for the periodic positives, 0 for the noisy negatives).
    """
    torch.manual_seed(seed)
    t = torch.arange(_L, dtype=torch.float32)
    tone = torch.sin(2 * np.pi * t / _TRUE_TAU)  # period == TRUE_TAU samples

    pos = []
    for _ in range(n_pos):
        phase = 2 * np.pi * torch.rand(1)
        ch0 = torch.sin(2 * np.pi * t / _TRUE_TAU + phase)
        win = torch.stack([ch0] + [0.05 * torch.randn(_L) for _ in range(3)], dim=-1)
        pos.append(win)
    neg = []
    for _ in range(n_neg):
        ch0 = torch.randn(_L)
        win = torch.stack([ch0] + [0.05 * torch.randn(_L) for _ in range(3)], dim=-1)
        neg.append(win)

    window = torch.stack(pos + neg, dim=0)  # [B, L, C]
    B = window.shape[0]
    labels = np.array([1] * n_pos + [0] * n_neg, dtype=np.float64)
    batch = {"window": window, "tau": torch.full((B,), _TRUE_TAU, dtype=torch.long)}
    # touch `tone` so the static analyzer keeps the analytic reference visible
    assert torch.allclose(tone[0], torch.tensor(0.0), atol=1e-6)
    return batch, labels


def _perwindow_regen_batch(n_pos: int = 16, n_neg: int = 16, seed: int = 1):
    """Batch spanning TWO operating points, each with its OWN tau.

    Group A windows are period-``_TAU_A`` tones (own tau ``_TAU_A``); group B are
    period-``_TAU_B`` tones (own tau ``_TAU_B``). Positives are periodic at their
    group's period, negatives are channel-0 noise. Each window carries its own
    ``tau`` plus the ``fs``/``tpf`` metadata from which ``round(fs / tpf)``
    reproduces that tau, so the sweep can derive each window's own delay.

    Returns ``(batch, labels)`` with ``batch`` keys ``window``/``tau``/``tpf``/``fs``.
    """
    torch.manual_seed(seed)
    t = torch.arange(_L, dtype=torch.float32)
    windows, taus, tpfs, labels = [], [], [], []
    for grp_tau in (_TAU_A, _TAU_B):
        tpf = _FS / grp_tau  # so round(fs / tpf) == grp_tau
        for _ in range(n_pos):
            phase = 2 * np.pi * torch.rand(1)
            ch0 = torch.sin(2 * np.pi * t / grp_tau + phase)
            win = torch.stack([ch0] + [0.05 * torch.randn(_L) for _ in range(3)], dim=-1)
            windows.append(win); taus.append(grp_tau); tpfs.append(tpf); labels.append(1)
        for _ in range(n_neg):
            ch0 = torch.randn(_L)
            win = torch.stack([ch0] + [0.05 * torch.randn(_L) for _ in range(3)], dim=-1)
            windows.append(win); taus.append(grp_tau); tpfs.append(tpf); labels.append(0)

    window = torch.stack(windows, dim=0)  # [B, L, C]
    B = window.shape[0]
    batch = {
        "window": window,
        "tau": torch.tensor(taus, dtype=torch.long),
        "tpf": torch.tensor(tpfs, dtype=torch.float32),
        "fs": torch.full((B,), _FS, dtype=torch.float32),
    }
    return batch, np.array(labels, dtype=np.float64)


# --------------------------------------------------------------------------- #
# 1. delay_ablation
# --------------------------------------------------------------------------- #
def test_delay_ablation_drops_auc_when_delay_removed():
    model = Regen()
    batch, labels = _regen_batch()
    res = physics.delay_ablation(model, batch, labels, big_tau=_BIG_TAU)
    assert set(res) >= {"auc_on", "auc_off", "auc_drop"}
    # With the real delay the periodic positives separate cleanly (high AUC);
    # zeroing the delayed term collapses every logit to ~0 (AUC ~ 0.5).
    assert res["auc_on"] > res["auc_off"]
    assert res["auc_off"] == pytest.approx(0.5, abs=0.1)
    assert res["auc_drop"] > 0.1


def test_delay_ablation_reports_mae_when_target_present():
    model = Regen()
    batch, labels = _regen_batch()
    batch["target"] = torch.as_tensor(labels, dtype=torch.float32)
    res = physics.delay_ablation(model, batch, labels, big_tau=_BIG_TAU)
    assert {"mae_on", "mae_off"} <= set(res)
    assert np.isfinite(res["mae_on"]) and np.isfinite(res["mae_off"])


def test_delay_ablation_omits_mae_without_target():
    model = Regen()
    batch, labels = _regen_batch()
    res = physics.delay_ablation(model, batch, labels, big_tau=_BIG_TAU)
    assert "mae_on" not in res and "mae_off" not in res


# --------------------------------------------------------------------------- #
# 2. tau_sweep
# --------------------------------------------------------------------------- #
def test_tau_sweep_peaks_at_true_tau():
    model = Regen()
    batch, labels = _regen_batch()
    taus = [_TRUE_TAU // 2, _TRUE_TAU, 2 * _TRUE_TAU, _BIG_TAU]
    res = physics.tau_sweep(model, batch, labels, taus)
    assert res["taus"] == taus
    assert len(res["auc"]) == len(taus)
    best = taus[int(np.argmax(res["auc"]))]
    assert best == _TRUE_TAU


# --------------------------------------------------------------------------- #
# 2b. tau_sweep_relative (per-window-relative sweep)
# --------------------------------------------------------------------------- #
def _expected_imposed(factor: float, own):
    """Reference imposition rule mirrored by the implementation."""
    return [max(1, int(round(factor * o))) for o in own]


def test_tau_sweep_relative_factor1_reproduces_each_windows_own_tau():
    model = PerWindowRegen()
    batch, labels = _perwindow_regen_batch()
    res = physics.tau_sweep_relative(model, batch, labels, factors=(0.5, 1.0, 2.0))

    assert set(res) >= {"factors", "auc", "own_tau", "imposed_tau", "baseline_auc"}
    assert res["factors"] == [0.5, 1.0, 2.0]
    assert len(res["auc"]) == len(res["factors"])

    # Each window's own tau is round(fs / tpf) -- here _TAU_A for group A, _TAU_B for B.
    expected_own = [int(round(float(fs) / float(tpf)))
                    for fs, tpf in zip(batch["fs"].tolist(), batch["tpf"].tolist())]
    assert res["own_tau"] == expected_own

    # The x1 factor imposes each window's OWN physical delay, per-window.
    i1 = res["factors"].index(1.0)
    assert res["imposed_tau"][i1] == res["own_tau"]


def test_tau_sweep_relative_is_per_window_not_global():
    model = PerWindowRegen()
    batch, labels = _perwindow_regen_batch()
    res = physics.tau_sweep_relative(model, batch, labels, factors=(0.5, 1.0, 2.0))

    own = res["own_tau"]
    # For every factor the imposed delay is that factor times each window's OWN
    # tau -- so windows with different own tau get different absolute delays.
    for f, imposed in zip(res["factors"], res["imposed_tau"]):
        assert imposed == _expected_imposed(f, own)
    i05 = res["factors"].index(0.5)
    # Two operating points -> at least two distinct imposed delays for one factor
    # (a *global* sweep would impose a single value on every window instead).
    assert len(set(res["imposed_tau"][i05])) >= 2


def test_tau_sweep_relative_baseline_matches_delay_ablation():
    model = PerWindowRegen()
    batch, labels = _perwindow_regen_batch()
    res = physics.tau_sweep_relative(model, batch, labels, factors=(1.0,), big_tau=_BIG_TAU)
    abl = physics.delay_ablation(model, batch, labels, big_tau=_BIG_TAU)
    # The no-delay (infinite-delay) baseline is exactly the ablation "off" path.
    assert res["baseline_auc"] == pytest.approx(abl["auc_off"], abs=1e-9)


def test_tau_sweep_relative_beats_a_single_global_delay():
    model = PerWindowRegen()
    batch, labels = _perwindow_regen_batch()
    rel = physics.tau_sweep_relative(model, batch, labels, factors=(1.0,))
    auc_rel = rel["auc"][rel["factors"].index(1.0)]

    # A single GLOBAL tau (group A's) mis-delays group B's windows, blurring the
    # peak; the per-window-relative x1 sweep aligns every window -> higher AUC.
    glob = physics.tau_sweep(model, batch, labels, [_TAU_A])
    assert auc_rel > glob["auc"][0]
    assert auc_rel > 0.9


def test_tau_sweep_relative_falls_back_to_batch_tau_without_fs_tpf():
    model = PerWindowRegen()
    batch, labels = _perwindow_regen_batch()
    stripped = {"window": batch["window"], "tau": batch["tau"]}  # no fs/tpf
    res = physics.tau_sweep_relative(model, stripped, labels, factors=(1.0,))
    assert res["own_tau"] == [int(v) for v in batch["tau"].tolist()]
    i1 = res["factors"].index(1.0)
    assert res["imposed_tau"][i1] == res["own_tau"]


# --------------------------------------------------------------------------- #
# 3. delayed_weight_share
# --------------------------------------------------------------------------- #
def _tkan(seed: int = 0) -> TKANModel:
    torch.manual_seed(seed)
    return TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)


def _share_batch(seed: int = 3):
    torch.manual_seed(seed)
    window = torch.randn(6, _L, 4)
    return {"window": window, "tau": torch.full((6,), _TRUE_TAU, dtype=torch.long)}


def test_delayed_weight_share_normalized():
    model = _tkan()
    batch = _share_batch()
    acts = collect_activations(model, batch)
    res = physics.delayed_weight_share(model, acts)
    assert set(res) >= {"overall", "per_gate"}
    overall = res["overall"]
    assert set(overall) == {"x_t", "h_prev", "x_delay"}
    assert all(np.isfinite(v) for v in overall.values())
    assert sum(overall.values()) == pytest.approx(1.0, abs=1e-6)


def test_delayed_weight_share_amplifying_delay_edges_raises_share():
    import copy

    model = _tkan()
    batch = _share_batch()
    base_acts = collect_activations(model, batch)
    base = physics.delayed_weight_share(model, base_acts)

    # Amplify the x_delay-block edges of one encoder gate, then re-measure.
    amp = copy.deepcopy(model)
    emap = edge_index_map(amp)
    delay_idx = [
        e["index"] for e in emap["encoder.layers.0.kan_z"] if e["block"] == "x_delay"
    ]
    assert delay_idx, "expected x_delay edges on the delay-aware encoder gate"
    with torch.no_grad():
        amp.encoder.layers[0].kan_z.base_weight[:, delay_idx] = 5.0
    amp_acts = collect_activations(amp, batch)
    amplified = physics.delayed_weight_share(amp, amp_acts)

    assert amplified["overall"]["x_delay"] > base["overall"]["x_delay"]


# --------------------------------------------------------------------------- #
# 4. attribution_spectrum
# --------------------------------------------------------------------------- #
def test_attribution_spectrum_peaks_near_f0():
    f0 = 1000.0  # Hz; lands exactly on FFT bin 4 (df = fs/L = 250 Hz)
    model = FreqStub(f0=f0, fs=_FS, length=_L)
    # Channel 0 = the strictly-positive envelope -> IG[t,0] = ref[t]**2 >= 0.
    window = np.zeros((_L, 4), dtype=np.float32)
    window[:, 0] = model.reference()
    res = physics.attribution_spectrum(
        model, window, tau=_TRUE_TAU, fs=_FS, tpf=_TPF, target="chatter_logit", steps=24
    )
    assert set(res) >= {"freqs", "power", "tpf", "tpf_band_energy", "total_energy"}
    freqs = np.asarray(res["freqs"])
    power = np.asarray(res["power"])
    assert freqs.shape == power.shape
    peak_freq = freqs[int(np.argmax(power))]
    df = freqs[1] - freqs[0]
    assert abs(peak_freq - f0) <= df  # nearest FFT bin to F0
    assert res["tpf"] == pytest.approx(_TPF)
    assert 0.0 <= res["tpf_band_energy"] <= res["total_energy"] + 1e-6


# --------------------------------------------------------------------------- #
# 4b. attribution_chatter_band (fn + tooth-pass sidebands, aggregated over windows)
# --------------------------------------------------------------------------- #
_FN_CB = 900.0   # natural / chatter frequency (Hz)
_TPF_CB = 200.0  # tooth-pass frequency (Hz) -- sideband spacing
_L_CB = 100      # df = fs / L = 100 Hz, so fn and fn+-tpf land exactly on bins


class Ch0Sum(nn.Module):
    """Linear passthrough on channel 0: ``logit = sum_t w[:, t, 0]``.

    For this model straight-path IG is exact, ``IG[t, 0] = w[t, 0]`` (zero baseline)
    and ``IG[t, c>0] = 0``, so the per-timestep attribution magnitude ``a[t]`` equals
    ``|w[t, 0]|`` -- letting a test place attribution energy at chosen frequencies.
    """

    def forward(self, batch: dict) -> dict:
        w = batch["window"]
        logit = w[:, :, 0].sum(dim=1)
        return {"chatter_logit": logit, "b_over_blim": logit}


def _cb_window(kind: str) -> np.ndarray:
    """A ``[L, 4]`` window whose channel-0 magnitude has a chosen line spectrum."""
    t = np.arange(_L_CB) / _FS
    if kind == "chatter":
        # Positive AM signal: carrier at fn plus tooth-pass sidebands fn +- tpf.
        ch0 = (2.0 + np.cos(2 * np.pi * _FN_CB * t)
               + 0.5 * np.cos(2 * np.pi * (_FN_CB - _TPF_CB) * t)
               + 0.5 * np.cos(2 * np.pi * (_FN_CB + _TPF_CB) * t))
    else:  # control: a single line at 300 Hz, far from fn and any sideband
        ch0 = 1.5 + np.cos(2 * np.pi * 300.0 * t)
    w = np.zeros((_L_CB, 4), dtype=np.float32)
    w[:, 0] = ch0
    return w


def test_attribution_chatter_band_high_for_chatter_low_for_control():
    model = Ch0Sum()
    windows = [_cb_window("chatter"), _cb_window("chatter"), _cb_window("control")]
    res = physics.attribution_chatter_band(
        model, windows, tau=_TRUE_TAU, fs=_FS, fn=_FN_CB, tpf=_TPF_CB,
        target="chatter_logit", steps=8, n_sidebands=2,
    )
    assert set(res) >= {
        "per_window_fraction", "chatter_fraction", "chatter_fraction_median",
        "band_centers", "bandwidth", "n_windows",
    }
    frac = np.asarray(res["per_window_fraction"])
    assert frac.shape == (3,)
    assert res["n_windows"] == 3
    # Chatter windows put nearly all attribution energy in the fn + sideband bands;
    # the control window (line at 300 Hz) puts almost none.
    assert frac[0] > 0.9 and frac[1] > 0.9
    assert frac[2] < 0.1
    # The centre band and both first sidebands are among the band centres.
    assert _FN_CB in res["band_centers"]
    assert (_FN_CB - _TPF_CB) in res["band_centers"]
    assert (_FN_CB + _TPF_CB) in res["band_centers"]


def test_attribution_chatter_band_aggregates_distribution():
    model = Ch0Sum()
    windows = [_cb_window("chatter"), _cb_window("control")]
    res = physics.attribution_chatter_band(
        model, windows, tau=_TRUE_TAU, fs=_FS, fn=_FN_CB, tpf=_TPF_CB, steps=8,
    )
    frac = np.asarray(res["per_window_fraction"])
    # Aggregates match the per-window distribution.
    assert res["chatter_fraction"] == pytest.approx(float(np.mean(frac)))
    assert res["chatter_fraction_median"] == pytest.approx(float(np.median(frac)))
    assert 0.0 <= res["chatter_fraction"] <= 1.0


def test_attribution_chatter_band_single_window_shape():
    model = Ch0Sum()
    res = physics.attribution_chatter_band(
        model, _cb_window("chatter"), tau=_TRUE_TAU, fs=_FS, fn=_FN_CB, tpf=_TPF_CB, steps=8,
    )
    assert res["n_windows"] == 1
    assert np.asarray(res["per_window_fraction"]).shape == (1,)


# --------------------------------------------------------------------------- #
# 5. figures
# --------------------------------------------------------------------------- #
def test_plot_tau_sweep_writes_file(tmp_path):
    model = Regen()
    batch, labels = _regen_batch()
    res = physics.tau_sweep(model, batch, labels, [_TRUE_TAU // 2, _TRUE_TAU, 2 * _TRUE_TAU])
    path = tmp_path / "tau_sweep.png"
    out = physics.plot_tau_sweep(res, str(path), true_tau=_TRUE_TAU)
    assert out == str(path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_tau_sweep_writes_file_relative(tmp_path):
    model = PerWindowRegen()
    batch, labels = _perwindow_regen_batch()
    res = physics.tau_sweep_relative(model, batch, labels, factors=(0.5, 1.0, 2.0))
    path = tmp_path / "tau_sweep_relative.png"
    out = physics.plot_tau_sweep(res, str(path))
    assert out == str(path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_attr_spectrum_writes_file(tmp_path):
    model = FreqStub(f0=1000.0, fs=_FS, length=_L)
    window = np.zeros((_L, 4), dtype=np.float32)
    window[:, 0] = model.reference()
    res = physics.attribution_spectrum(model, window, tau=_TRUE_TAU, fs=_FS, tpf=_TPF, steps=16)
    path = tmp_path / "spectrum.png"
    out = physics.plot_attr_spectrum(res, str(path))
    assert out == str(path)
    assert path.exists() and path.stat().st_size > 0
