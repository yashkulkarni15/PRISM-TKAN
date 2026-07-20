"""Eval-only physics-discovery analyses for Phase 3 (no neural network).

Currently hosts :mod:`pri2m.analysis.sindy_dde`, a hand-rolled delay-embedded
sparse regression (SINDy-DDE) that recovers the regenerative delay ``tau`` and a
delayed-differential regenerative term straight from raw cutting signals -- an
independent white-box cross-check for the tooth-pass period and the TKAN's
delay-ablation result. Pure ``numpy``/``sklearn``; ``pysindy`` is NOT used.
"""

__all__ = ["sindy_dde"]
