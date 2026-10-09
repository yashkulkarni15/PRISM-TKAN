"""Sixteen per-window formula terms (4 per channel) for distilling a sparse equation.

Per channel c: ``D`` = log mean (x - x_tau)^2, ``S`` = log mean (x + x_tau)^2 over
``t >= tau_p`` (the delay-difference / delay-sum energies), ``H`` = log power in
bins at tooth-passing harmonics, ``C`` = log power in non-harmonic bins above
``0.5 * tpf`` (the chatter band). Every log is ``log(value + 1e-8)``.
"""
import numpy as np
import torch

from pri2m.models.minirocket_baseline import build_inputs
from pri2m.models.tkan_model import scale_tau

CHANNELS = ("xtpos", "ytpos", "Forcex", "Forcey")
TERM_NAMES = [f"{k}_{c}" for c in CHANNELS for k in ("D", "S", "H", "C")]
EPS = 1e-8


def window_terms(windows, tau_raw, tpf, fs, pool: int = 4) -> np.ndarray:
    """``[N, 16]`` float64 terms in ``TERM_NAMES`` order.

    ``windows`` are z-scored ``[N, 1024, 4]``; ``tau_raw`` is in RAW samples;
    ``tpf`` and ``fs`` are per-window Hz. A delay that reaches past the pooled
    window (``tau_p >= Lp``) leaves no valid samples, so D and S use mean 0.
    """
    tau_raw, tpf, fs = (np.asarray(a, dtype=np.float64).reshape(-1) for a in (tau_raw, tpf, fs))
    X = build_inputs(windows, tau_raw, pool, "B").astype(np.float64)
    C = len(CHANNELS)
    x, xt = X[:, :C], X[:, C:]  # [N, C, Lp]
    Lp = x.shape[-1]

    tau_p = scale_tau(torch.as_tensor(tau_raw.astype(np.int64)), pool).numpy()
    valid = (np.arange(Lp)[None, :] >= tau_p[:, None])[:, None, :]  # [N, 1, Lp]
    n_valid = np.maximum(valid.sum(-1), 1)
    D = (valid * (x - xt) ** 2).sum(-1) / n_valid  # [N, C]
    S = (valid * (x + xt) ** 2).sum(-1) / n_valid

    P = np.abs(np.fft.rfft(x, axis=-1)) ** 2 / Lp**2  # [N, C, F]
    f = np.fft.rfftfreq(Lp, d=pool / fs[:, None])  # [N, F]
    df = (fs / pool / Lp)[:, None]
    k = np.maximum(1, np.round(f / tpf[:, None]))
    harm = np.abs(f - k * tpf[:, None]) <= np.maximum(0.02 * k * tpf[:, None], df / 2)
    chat = ~harm & (f > 0.5 * tpf[:, None])
    H = (P * harm[:, None]).sum(-1)
    Cb = (P * chat[:, None]).sum(-1)

    return np.log(np.stack([D, S, H, Cb], axis=-1) + EPS).reshape(len(x), -1)
