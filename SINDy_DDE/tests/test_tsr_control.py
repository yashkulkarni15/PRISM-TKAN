"""U12-B: does the TSR delay-lock come from learning, or from construction?

The 2026-07-23/24 run found that when TSR resolves a peak at all, the peak lands
on that window's own tau far beyond chance (34% exact, p = 0.00005). That is only
evidence about the *model* if it is not an artifact of how the input is built:
the delayed channel is literally the signal shifted by tau, so a lag-tau bump in
the relevance autocorrelation could arise from the construction alone, with no
learning involved.

The decisive control is an **untrained network**. Same architecture, same input
construction, random weights. If a randomly-initialised net locks to tau just as
hard, the lock says nothing about what was learned and the attribution view of
the delay is genuinely dead. A second condition -- the trained net fed shuffled
taus -- tests whether the lock follows whatever delay is injected.

These tests pin the *mechanics* of the control on synthetic models: that the
conditions are actually different, that the verdict rule is applied as written,
and that a construction artifact would be caught rather than explained away.
They do not assert a scientific outcome; that comes from the real run.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

from pri2m.models.tkan_model import TKANModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import whitebox_phase3 as W  # noqa: E402


def _model(seed: int) -> TKANModel:
    torch.manual_seed(seed)
    return TKANModel(
        input_size=4, hidden_size=8, num_layers=1, grid_size=5, spline_order=3
    ).eval()


def _batch(B=4, L=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(L, dtype=torch.float32)
    base = torch.sin(2 * np.pi * 0.05 * t)
    window = base.view(1, L, 1).repeat(B, 1, 4) + 0.1 * torch.randn(B, L, 4, generator=g)
    labels = np.array([0, 1] * (B // 2))
    return {
        "window": window,
        "tau": torch.tensor([5, 9, 13, 17][:B], dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float32),
        "b_over_blim": torch.rand(B, generator=g),
        "omega": torch.full((B,), 5000.0),
        "tpf": torch.full((B,), 200.0),
        "fs": torch.full((B,), 10_000.0),
    }, labels


# ---- tau_override: the mechanism the shuffled-tau condition needs ---------- #

def test_tsr_one_window_uses_the_batch_tau_by_default():
    batch, labels = _batch()
    row = W.tsr_one_window(_model(0), batch, None, labels, 1, steps=2)
    assert row["tau_raw"] == int(batch["tau"][1])


def test_tsr_one_window_honours_a_tau_override():
    """Without this the shuffled-tau condition cannot exist."""
    batch, labels = _batch()
    row = W.tsr_one_window(_model(0), batch, None, labels, 1, steps=2, tau_override=31)
    assert row["tau_raw"] == 31


# ---- the control itself --------------------------------------------------- #

def test_control_runs_all_three_conditions():
    batch, labels = _batch()
    out = W.run_tsr_control(_model(0), _model(99), batch, None, labels,
                            indices=[0, 1], steps=2, seed=0)
    assert set(out["conditions"]) == {
        "trained_true_tau", "untrained_true_tau", "trained_shuffled_tau"
    }
    for cond in out["conditions"].values():
        assert cond["n_windows"] == 2
        assert "resolved_fraction" in cond and "lock_z_mean" in cond


def test_shuffled_condition_actually_changes_the_taus():
    """A shuffle that returns the identity would silently make the control vacuous."""
    batch, labels = _batch()
    out = W.run_tsr_control(_model(0), _model(99), batch, None, labels,
                            indices=[0, 1, 2, 3], steps=2, seed=0)
    true_taus = [r["tau_raw"] for r in out["conditions"]["trained_true_tau"]["per_window"]]
    shuf_taus = [r["tau_raw"] for r in out["conditions"]["trained_shuffled_tau"]["per_window"]]
    assert sorted(shuf_taus) == sorted(true_taus)     # a permutation
    assert shuf_taus != true_taus                     # ...and not the identity


def test_verdict_flags_an_artifact_when_the_untrained_net_locks_just_as_hard():
    verdict = W.tsr_control_verdict(trained_lock=2.0, untrained_lock=1.9, threshold=0.8)
    assert verdict["artifact_suspected"] is True
    assert verdict["lock_gain_over_untrained"] < 0.2


def test_verdict_clears_the_lock_when_training_is_what_produces_it():
    # n_lock_tested is supplied because a clear now requires it, and because the
    # real caller supplies it (whitebox_phase3.py:510). Omitting it here was
    # what let the earlier version return a definitive clear on no stated
    # evidence at all.
    verdict = W.tsr_control_verdict(trained_lock=3.0, untrained_lock=0.2,
                                    threshold=0.8, n_lock_tested=24)
    assert verdict["artifact_suspected"] is False
    assert verdict["lock_gain_over_untrained"] > 2.5


def test_verdict_treats_a_missing_lock_as_undecided_not_as_a_pass():
    """No measurement must never read as 'cleared'."""
    v = W.tsr_control_verdict(trained_lock=None, untrained_lock=0.1, threshold=0.8)
    assert v["artifact_suspected"] is None
    v2 = W.tsr_control_verdict(trained_lock=2.0, untrained_lock=None, threshold=0.8)
    assert v2["artifact_suspected"] is None


def test_verdict_flags_an_artifact_when_the_untrained_net_locks_harder():
    v = W.tsr_control_verdict(trained_lock=0.5, untrained_lock=2.5, threshold=0.8)
    assert v["artifact_suspected"] is True


# --- the three checks the rule as *written* requires (added 2026-08-02) --------
#
# U12 section 4 recorded that the implementation was the ratio test alone, and
# that three things the pre-committed rule depends on were missing. A "cleared"
# verdict is the dangerous outcome, because it promotes TSR to a fourth line of
# delay evidence. These pin that a clear can only be reached on evidence that
# supports one. UNDECIDED is never a pass.


def test_a_trained_lock_that_is_not_a_lock_cannot_clear():
    """A trained value indistinguishable from noise must not read as learned.

    Ratio test alone: untrained 0.05 < 0.8 * 0.10, so the old rule returned
    False and printed the fourth-line-of-evidence claim off two non-locks.
    """
    v = W.tsr_control_verdict(trained_lock=0.10, untrained_lock=0.05,
                              threshold=0.8, n_lock_tested=24,
                              lock_z_threshold=2.0)
    assert v["artifact_suspected"] is None
    assert v["undecided_reason"] == "trained_lock_not_significant"


def test_a_trained_lock_pointing_the_wrong_way_cannot_clear():
    v = W.tsr_control_verdict(trained_lock=-3.0, untrained_lock=0.1,
                              threshold=0.8, n_lock_tested=24)
    assert v["artifact_suspected"] is None


def test_one_window_cannot_carry_a_definitive_verdict():
    """Under-powered is undecided in BOTH directions, not just the clear."""
    cleared = W.tsr_control_verdict(trained_lock=3.0, untrained_lock=0.2,
                                    threshold=0.8, n_lock_tested=1)
    assert cleared["artifact_suspected"] is None
    assert cleared["undecided_reason"] == "insufficient_windows"
    flagged = W.tsr_control_verdict(trained_lock=2.0, untrained_lock=1.9,
                                    threshold=0.8, n_lock_tested=1)
    assert flagged["artifact_suspected"] is None


def test_a_clear_requires_the_lock_to_be_specific_to_this_windows_tau():
    """The shuffled arm was computed at full cost and then ignored.

    A lock that is no stronger at the true tau than at a permuted one is
    reporting the smoothness of the attribution curve, not the delay.
    """
    v = W.tsr_control_verdict(trained_lock=3.0, untrained_lock=0.2,
                              threshold=0.8, n_lock_tested=24,
                              shuffled_lock=3.1)
    assert v["artifact_suspected"] is None
    assert v["undecided_reason"] == "not_tau_specific"
    still_clear = W.tsr_control_verdict(trained_lock=3.0, untrained_lock=0.2,
                                        threshold=0.8, n_lock_tested=24,
                                        shuffled_lock=0.9)
    assert still_clear["artifact_suspected"] is False


def test_the_shuffled_arm_is_recorded_even_when_it_does_not_change_the_verdict():
    v = W.tsr_control_verdict(trained_lock=2.81, untrained_lock=4.62,
                              threshold=0.8, n_lock_tested=24,
                              shuffled_lock=0.88)
    assert v["shuffled_lock"] == 0.88
    assert v["tau_specific"] is True


def test_an_unknown_window_count_cannot_clear_the_lock():
    """Same principle the lock arms already follow, applied to the evidence count.

    ``min_windows`` was skipped entirely when ``n_lock_tested`` was not supplied,
    so a caller that forgot it got a definitive clear with nothing establishing
    how much evidence stood behind it. A missing measurement must read as
    UNDECIDED, never as a pass.
    """
    v = W.tsr_control_verdict(trained_lock=3.0, untrained_lock=0.2,
                              shuffled_lock=0.1, n_lock_tested=None)

    assert v["artifact_suspected"] is None
    assert v["undecided_reason"] == "window_count_unknown"


def test_an_unknown_window_count_still_permits_the_artifact_call():
    """The conservative outcome does not need the count to stand."""
    v = W.tsr_control_verdict(trained_lock=2.0, untrained_lock=1.9,
                              n_lock_tested=None)

    assert v["artifact_suspected"] is True


def test_a_shuffled_arm_just_below_the_trained_one_is_not_tau_specific():
    """A hairline gap is not specificity.

    ``tau_specific`` was a bare ``trained > shuffled``, so a shuffled lock a
    ten-thousandth below the trained one satisfied it and permitted a clear. The
    shuffled arm now has to sit as far below the trained lock as the untrained
    arm does -- the same ratio, rather than a second invented constant.
    """
    v = W.tsr_control_verdict(trained_lock=3.0, untrained_lock=0.2,
                              shuffled_lock=2.9999, n_lock_tested=24)

    assert v["tau_specific"] is False
    assert v["artifact_suspected"] is None
    assert v["undecided_reason"] == "not_tau_specific"


def test_a_genuinely_specific_shuffled_arm_still_reads_as_specific():
    """The real fold2_w512 values must keep their recorded reading."""
    v = W.tsr_control_verdict(trained_lock=2.811570240129822,
                              untrained_lock=0.2,
                              shuffled_lock=0.8782125023974482,
                              n_lock_tested=24)

    assert v["tau_specific"] is True


def test_returned_run_gives_the_same_verdict_under_the_corrected_rule():
    """Regression pin on the real fold2_w512 result (GL job 55408255).

    Values are from results/phase3/xai/tsr_control/fold2_w512/
    tsr_control_manifest.json, provenance.complete = true. U12 section 4 states
    that applying the rule as written changes nothing here. This is that claim,
    executable.
    """
    v = W.tsr_control_verdict(
        trained_lock=2.811570240129822,      # lock_z_mean, trained_true_tau
        untrained_lock=4.615321031499511,    # lock_z_mean, untrained_true_tau
        threshold=0.8,
        shuffled_lock=0.8782125023974482,    # lock_z_mean, trained_shuffled_tau
        n_lock_tested=24,
    )
    assert v["artifact_suspected"] is True
    assert v["undecided_reason"] is None


def test_control_result_carries_the_verdict():
    batch, labels = _batch()
    out = W.run_tsr_control(_model(0), _model(99), batch, None, labels,
                            indices=[0, 1], steps=2, seed=0)
    assert "verdict" in out
    assert set(out["verdict"]) >= {
        "trained_lock", "untrained_lock", "lock_gain_over_untrained",
        "artifact_suspected",
    }


def test_control_does_not_mutate_the_batch():
    batch, labels = _batch()
    before = batch["window"].clone()
    before_tau = batch["tau"].clone()
    W.run_tsr_control(_model(0), _model(99), batch, None, labels,
                      indices=[0, 1], steps=2, seed=0)
    np.testing.assert_allclose(batch["window"].numpy(), before.numpy())
    np.testing.assert_array_equal(batch["tau"].numpy(), before_tau.numpy())


# ---- surviving a wall-clock kill ------------------------------------------ #
# Job 55098053 (2026-07-26) hit its 12h wall having written nothing and printed
# nothing since the header, so twelve hours of compute left no evidence and no
# way to tell a slow run from a hung one. Both are fixed here.

def test_control_reports_progress_for_every_window_it_scores():
    """A silent log must never be indistinguishable from a hang."""
    batch, labels = _batch()
    seen = []
    W.run_tsr_control(_model(0), _model(99), batch, None, labels,
                      indices=[0, 1], steps=2, seed=0,
                      progress=lambda *a: seen.append(a))

    assert len(seen) == 6                       # 3 conditions x 2 windows
    names = [s[0] for s in seen]
    assert names.count("trained_true_tau") == 2
    assert [s[1] for s in seen[:2]] == [1, 2]   # 1-based done counter
    assert all(s[2] == 2 for s in seen)         # total per condition


def test_control_hands_back_a_partial_result_after_each_condition():
    """The hook the runner writes partial manifests from."""
    batch, labels = _batch()
    partials = []
    W.run_tsr_control(_model(0), _model(99), batch, None, labels,
                      indices=[0, 1], steps=2, seed=0,
                      on_condition=lambda name, res: partials.append((name, res)))

    assert [p[0] for p in partials] == [
        "trained_true_tau", "untrained_true_tau", "trained_shuffled_tau"
    ]
    # Each hand-off carries every condition finished so far, and nothing else.
    assert [len(p[1]["conditions"]) for p in partials] == [1, 2, 3]
    assert "trained_shuffled_tau" not in partials[1][1]["conditions"]


def test_a_partial_result_is_decidable_once_the_decisive_pair_is_done():
    """A kill during the third condition must still leave a usable verdict.

    trained + untrained ARE the control; trained_shuffled_tau only supports it.
    """
    batch, labels = _batch()
    partials = []
    W.run_tsr_control(_model(0), _model(99), batch, None, labels,
                      indices=[0, 1], steps=2, seed=0,
                      on_condition=lambda name, res: partials.append(res))

    assert partials[0]["verdict"] is None            # one side only: not decidable
    assert partials[1]["verdict"] is not None        # decisive pair complete
    assert set(partials[1]["verdict"]) >= {"trained_lock", "untrained_lock",
                                           "artifact_suspected"}


def test_the_decisive_pair_is_scored_before_the_supporting_condition():
    """Load-bearing ordering: it is what makes a timeout survivable."""
    batch, labels = _batch()
    order = []
    W.run_tsr_control(_model(0), _model(99), batch, None, labels,
                      indices=[0], steps=2, seed=0,
                      on_condition=lambda name, res: order.append(name))
    assert order.index("trained_shuffled_tau") == 2


def test_progress_and_partial_hooks_are_optional():
    """Omitting both must reproduce the old result exactly."""
    batch, labels = _batch()
    plain = W.run_tsr_control(_model(0), _model(99), batch, None, labels,
                              indices=[0, 1], steps=2, seed=0)
    hooked = W.run_tsr_control(_model(0), _model(99), batch, None, labels,
                               indices=[0, 1], steps=2, seed=0,
                               progress=lambda *a: None,
                               on_condition=lambda *a: None)
    assert plain["verdict"] == hooked["verdict"]
    assert set(plain["conditions"]) == set(hooked["conditions"])


def test_provenance_assembles_without_blowing_up():
    """Job 55405355 died here, AFTER the science, on `_git_sha()` missing its root.

    Every other runner calls `_git_sha(_ROOT)`; this one alone called it bare, so
    the manifest step raised TypeError. The old code only reached that line at
    the very end, so the 12h timeout hid it -- the run has to survive provenance.
    """
    import tsr_control_phase3 as T

    prov = T._provenance(
        checkpoint="x.pt", run_id="abc", tag="fold2", n_sampled=64, n_scored=24,
        tsr_steps=4, sample_seed=0, shuffle_seed=0, init_seed=0,
        n_reset=7, mean_weight_diff=0.426,
    )
    assert prov["checkpoint"] == "x.pt"
    assert prov["n_windows_scored"] == 24
    assert "git_sha" in prov            # None is fine; raising is not
    assert "rendered_date" in prov


def test_a_partial_manifest_is_marked_incomplete():
    """A killed run must never be readable as a finished one."""
    import tsr_control_phase3 as T

    prov = {"checkpoint": "x.pt", "tag": "fold2"}
    result = {"conditions": {"trained_true_tau": {}}, "verdict": None}

    partial = T._manifest(result, prov, complete=False)
    final = T._manifest(result, prov, complete=True)

    assert partial["provenance"]["complete"] is False
    assert final["provenance"]["complete"] is True
    assert partial["control"] is result
    assert partial["provenance"]["checkpoint"] == "x.pt"   # provenance preserved


def test_feature_masking_does_not_change_the_lag_statistics():
    """The 3x speedup the control relies on must be exactly free for lag stats.

    feature_masking selects how TSR scores the FEATURE dimension in step 2, while
    tr_peak_lag / tr_lock_z / tr_peak_at_floor all derive from step 1's
    time_relevance. If that ever stops holding, the control's cheap setting would
    silently be measuring something else than the sweep did.
    """
    batch, labels = _batch()
    model = _model(0)
    full = W.tsr_one_window(model, batch, None, labels, 1, steps=2, feature_masking=True)
    cheap = W.tsr_one_window(model, batch, None, labels, 1, steps=2, feature_masking=False)

    assert cheap["tr_peak_lag"] == full["tr_peak_lag"]
    assert cheap["tr_peak_at_floor"] == full["tr_peak_at_floor"]
    if full["tr_lock_z"] is None:
        assert cheap["tr_lock_z"] is None
    else:
        assert abs(cheap["tr_lock_z"] - full["tr_lock_z"]) < 1e-9
