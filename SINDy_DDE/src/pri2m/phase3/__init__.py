"""Phase 3 data preparation: leakage-safe splits, label hygiene, sample weights.

Built 2026-06-03 to close the pre-Phase-3 audit findings before MLP/TKAN-PINN:
  - mirror leakage (clean D shares 100% of operating points with noisy D+60/D+90)
  - 700 negative-blim rows carrying untrustworthy unstable labels
  - heavy-tailed noise_score and NaN metrics_unavailable rows used as sample weights
"""
