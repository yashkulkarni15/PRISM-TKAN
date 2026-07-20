<!--
  TEMPLATE — Phase 3 TKAN white-box analysis report.

  This file is a TEMPLATE. `scripts/render_whitebox_report.py` reads
  `whitebox_manifest.json` (produced by `scripts/whitebox_phase3.py`) and
  substitutes every `{{placeholder}}` with a REAL value, writing the rendered
  report next to the manifest. No number in the rendered report is invented:
  if a value is missing from the manifest, the renderer writes `N/A (not run)`
  rather than a guess.

  Placeholder inventory (manifest keys the renderer must supply):
    provenance: checkpoint, run_id, tag, git_sha, rendered_date, n_windows,
                n_channels, hidden_size, num_layers, grid_size, spline_order,
                side_input, side_features, tau_samples, fs, tpf, val_auc
    features:   chan_importance_table, top_channel, side_importance_table,
                fig_channel_importance, fig_lag_profile, lag_peak_samples
    how:        n_edges_total, index_map_table, fig_edge_activations
    why:        delay_auc_on, delay_auc_off, delay_auc_drop,
                tau_sweep_table, tau_argmax, tau_physical,
                weight_share_xdelay, weight_share_xt, weight_share_hprev,
                fig_tau_sweep, fig_attr_spectrum, tpf_band_energy_frac,
                tsr_table, tsr_delay_share_gain, tsr_windows, tsr_steps
    bsplines:   symbolic_table, n_symbolic, frac_linear, frac_nonlinear,
                fig_top_edges, fig_spline_atlas, fig_feature_bars,
                fig_edge_importance_named, atlas_table,
                n_atlas_delay_edges, n_atlas_side_edges
    faithful:   prune_table, prune_auc_full, prune_auc_5pct,
                deletion_drop, randomization_cosine,
                fig_pruning_curve, fig_deletion_curve
    cases:      case_chatter, case_stable, fig_case_chatter, fig_case_stable
    caveats:    caveats_list
-->

# White-Box Analysis of the Phase 3 TKAN Milling-Chatter Model

**Checkpoint:** `{{checkpoint}}` (run `{{run_id}}`, tag `{{tag}}`) ·
**Validation AUC:** {{val_auc}} · **Rendered:** {{rendered_date}} · **Git:** `{{git_sha}}`
Windows analysed: {{n_windows}} · Model: {{n_channels}}-channel input, hidden {{hidden_size}},
{{num_layers}} layer(s), B-spline grid {{grid_size}}/order {{spline_order}},
side-input {{side_input}} ({{side_features}}), delay τ = {{tau_samples}} samples
(fs = {{fs}} Hz, tooth-pass freq = {{tpf}} Hz).

> **What "white box" means here.** A Kolmogorov–Arnold network replaces every dense
> weight with an explicit, plottable 1-D function on each input→output edge. So unlike
> an MLP, this model's computation can be read directly: which physical signals it
> consumes, the closed-form shape of each learned activation, and how a single decision
> is assembled. This report makes that reading concrete and, where possible, checks the
> explanations against the model's own behaviour rather than asserting them.

---

## 1. What the model is

The model maps a short multi-channel cutting-signal window to two outputs: a severity
regression `b_over_blim = b / blim` (axial depth over the stability limit; ≥ 1 means
chatter) and a binary chatter logit. It is a **Temporal KAN**: a GRU-style recurrence
whose three gate maps are `KANLinear` layers (per-edge cubic B-spline + SiLU base),
walked over the window, feeding a KAN severity head.

Three design choices make it physically motivated rather than a generic sequence model:

1. **An explicit kinematic delay.** At each step the bottom recurrent layer is fed not
   only the current sample but the sample from **τ = round(fs / tpf)** steps earlier —
   the tooth-pass period. This mirrors the *regenerative effect*: the chip thickness a
   tooth removes depends on the wavy surface left by the previous tooth one period ago.
   Regenerative chatter is classically a **delay differential equation (DDE)**; the
   delay is wired into the architecture.
2. **KAN edges instead of dense weights**, so every learned interaction is a readable
   1-D function (§5).
3. **Process parameters as a side input** (spindle speed, tooth-pass frequency, the
   delay), following Gao et al. — fed to the severity head only.

The total learned object is **{{n_edges_total}} one-dimensional edge functions**. The
rest of this report reads them.

---

## 2. Features — what the model reads

Per-window integrated-gradients attribution, aggregated over {{n_windows}} windows and
split by class, ranks the input channels by how much they move the chatter decision:

{{chan_importance_table}}

The most influential channel is **{{top_channel}}**.

![Channel importance, chatter vs stable]({{fig_channel_importance}})

**Time structure.** Averaging attribution over channels gives a per-lag profile — *when*
in the window the evidence sits:

![Per-lag attribution]({{fig_lag_profile}})

The profile's peak sits near lag {{lag_peak_samples}} samples (cf. the physical delay
τ = {{tau_samples}}); §4 tests the delay dependence directly.

**Process-parameter side features.** Permuting each side feature and measuring the AUC
drop:

{{side_importance_table}}

---

## 3. How — the mechanism

Each recurrent step computes an update gate `z`, a reset gate `r`, and a candidate state,
every one a KAN map of `cat([x_t, h_prev, x_delay])`. The raw indices into that vector
are opaque on their own; the semantic index map makes them legible:

{{index_map_table}}

This is the key to reading the gates: an edge in the **`x_delay`** block reads the
τ-delayed channel — the regenerative term — while the **`x_t`** block reads the current
sample and **`h_prev`** the recurrent hidden state. The learned edge activations
(base + spline) themselves are read in §5; the
data-aware importance of every edge, per KANLinear, is mapped here:

![Per-KANLinear edge-importance heatmap]({{fig_edge_activations}})

---

## 4. Why — physics linkage (the regenerative delay)

If the model has learned the regenerative mechanism, three things should hold: removing
the delayed input should hurt; performance should be best at the *physical* delay; and a
real share of the gate computation should rest on the delayed channels. All three are
measured, not assumed.

**Delay ablation.** Forcing the delayed input to zero (so the recurrence sees only the
current sample and its own state) changes chatter AUC from **{{delay_auc_on}}** to
**{{delay_auc_off}}** — a drop of **{{delay_auc_drop}}**.

**τ-sweep.** Re-running the model with the delay set to a range of values:

{{tau_sweep_table}}

AUC is maximised at τ = {{tau_argmax}} samples; the physical tooth-pass delay is
{{tau_physical}}.

![AUC vs delay]({{fig_tau_sweep}})

**Where the gate's attention rests.** Summing the data-aware edge importance by input
block across the encoder gates:

- τ-delayed channels (`x_delay`): **{{weight_share_xdelay}}**
- current channels (`x_t`): {{weight_share_xt}}
- hidden state (`h_prev`): {{weight_share_hprev}}

**Spectral fingerprint.** The FFT of the time-resolved attribution concentrates
**{{tpf_band_energy_frac}}** of its energy in a band around the tooth-pass frequency:

![Attribution spectrum]({{fig_attr_spectrum}})

### 4b. Temporal Saliency Rescaling

TSR separates time relevance from feature relevance over {{tsr_windows}} windows using
{{tsr_steps}} integration steps:

{{tsr_table}}

The mean channel-share change from raw IG to TSR is {{tsr_delay_share_gain}}. TSR is a
channel/time-resolved saliency measure: the delayed channels' share is the observable
here, while the tau input's causal role is established by the intervention above, not
by TSR alone.

---

## 5. The B-splines — the learned edge functions

The white-box core: each important edge's learned activation φ(x) = base·SiLU(x) +
spline(x), fitted to a closed form over the range the data actually visits, ranked by
data-aware importance.

{{symbolic_table}}

Of the {{n_symbolic}} top edges, {{frac_linear}} are well-described by a linear form and
{{frac_nonlinear}} are genuinely nonlinear (saturating / oscillatory / higher-order) —
the part of the computation a linear model could not capture.

![Top edge activations with fitted forms]({{fig_top_edges}})

### 5b. Spline-edge atlas

The atlas ranks and names the strongest learned edge functions. It contains
{{n_atlas_delay_edges}} delayed-input edges and {{n_atlas_side_edges}} side-input edges
among the displayed set:

{{atlas_table}}

![Named spline-edge atlas]({{fig_spline_atlas}})

![Feature-level edge importance]({{fig_feature_bars}})

![Named edge-importance heatmaps]({{fig_edge_importance_named}})

---

## 6. Faithfulness — do the explanations hold up?

**Edge pruning.** Keeping only the top-importance edges and zeroing the rest, then
measuring AUC:

{{prune_table}}

Full model AUC {{prune_auc_full}}; keeping only the top 5% of edges gives
{{prune_auc_5pct}} — a measure of how concentrated the decision is.

![Pruning curve]({{fig_pruning_curve}})

**Attribution deletion.** Progressively deleting the highest-attributed input cells
drives the prediction by {{deletion_drop}}.

![Deletion curve]({{fig_deletion_curve}})

**Sanity check.** Attribution on the trained model vs a re-initialised copy has cosine
similarity {{randomization_cosine}} — low similarity confirms the attributions reflect
*learned* structure, not architecture alone.

---

## 7. Case studies — two decisions, walked through

**A confident chatter window:**

{{case_chatter}}

![Chatter case walkthrough]({{fig_case_chatter}})

**A confident stable window:**

{{case_stable}}

![Stable case walkthrough]({{fig_case_stable}})

---

## 8. Caveats and honesty

{{caveats_list}}

- All numbers above are computed from the named checkpoint on the sampled windows; none
  are illustrative. Where an analysis was not run, the field reads `N/A (not run)`.
- Attribution and importance are *model-faithful* explanations, not ground-truth
  causation in the physics; §4/§6 check them against the model's own behaviour but cannot
  prove the underlying mechanism.
- The headline classification AUC is validation-selected; the held-out generalization
  figure (datasets 121–125) is the honest out-of-sample number — see
  `docs/phase3_tkan/heldout_eval_121_155.md`.
- The severity (`b_over_blim`) head is the known-weak head (heavy-tailed error); treat
  its attributions as more tentative than the chatter head's.
