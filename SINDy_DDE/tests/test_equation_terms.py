import numpy as np

from pri2m.xai.equation_terms import CHANNELS, TERM_NAMES, window_terms

FLOOR = np.log(1e-8)
N, L, FS, TPF = 3, 1024, 4096.0, 64.0  # pooled rate 1024 Hz, df = 4 Hz, period = 64 raw samples


def tone_fixture(freq_hz, tau_raw):
    t = np.arange(L) / FS
    ph = np.array([0.0, 1.0, 2.0])[:, None]
    w = np.sin(2 * np.pi * freq_hz * t[None, :] + ph)[:, :, None].repeat(4, axis=2)
    n = lambda v: np.full(N, v, dtype=float)
    return w, n(tau_raw), n(TPF), n(FS)


def test_term_names_order():
    assert TERM_NAMES[:4] == ["D_xtpos", "S_xtpos", "H_xtpos", "C_xtpos"] and len(TERM_NAMES) == 16
    assert CHANNELS == ("xtpos", "ytpos", "Forcex", "Forcey")


def test_identical_delay_floors_D():
    w, tau, tpf, fs = tone_fixture(TPF, tau_raw=64)  # tau == period -> x_tau == x
    T = window_terms(w, tau, tpf, fs)
    assert T.shape == (N, 16) and T.dtype == np.float64
    assert np.all(T[:, TERM_NAMES.index("D_ytpos")] < FLOOR + 1.0)


def test_antiphase_floors_S():  # half-period delay on a pure tone -> x_tau == -x
    w, tau, tpf, fs = tone_fixture(TPF, tau_raw=32)
    T = window_terms(w, tau, tpf, fs)
    assert np.all(T[:, TERM_NAMES.index("S_ytpos")] < FLOOR + 1.0)
    assert np.all(T[:, TERM_NAMES.index("D_ytpos")] > FLOOR + 5.0)


def test_tone_at_harmonic_lands_in_H_between_lands_in_C():
    for f, h_wins in ((2 * TPF, True), (2.5 * TPF, False)):
        T = window_terms(*tone_fixture(f, 64))
        H, C = (T[:, TERM_NAMES.index(f"{k}_xtpos")] for k in "HC")
        assert np.all(H > C) if h_wins else np.all(C > H)


def test_tau_beyond_window_is_finite():  # Review Focus 1
    w, _, tpf, fs = tone_fixture(TPF, 64)
    T = window_terms(w, np.full(N, 4 * 300), tpf, fs)
    assert np.isfinite(T).all()
    assert np.allclose(T[:, TERM_NAMES.index("D_xtpos")], FLOOR)


def test_no_bins_below_nyquist_gives_floor():  # Review Focus 2
    w, tau, _, fs = tone_fixture(TPF, 64)
    T = window_terms(w, tau, np.full(N, 1e9), fs)
    assert np.allclose(T[:, TERM_NAMES.index("H_xtpos")], FLOOR)
    assert np.allclose(T[:, TERM_NAMES.index("C_xtpos")], FLOOR)
