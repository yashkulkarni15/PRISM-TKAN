# Phase 3 TKAN — verified external-research scan (2026-06)

Four parallel web-research subagents, **verified-only** protocol (every item fetched from a
real page with a confirming quote; nothing cited from memory). Items the agents could not
fetch are tagged. The **★-starred post-cutoff papers were independently re-fetched by the
driver** (title/authors/abstract-quote confirmed), not relayed on trust.

Verification tags: **[FETCHED]** read directly · **[SEARCH-VERIFIED]** metadata confirmed
across indexes but page was auth-gated · **★ re-verified** by the driver.

---

## A. KAN interpretability & symbolic extraction (improves `symbolic.py`)

- **In-Context Symbolic Regression for Robustness-Improved KANs** — Sovrano, Losavio, Vilone,
  Langheinrich, 2026. arXiv:2603.15250. **[FETCHED] ★** — *"the standard KAN-to-symbol approach
  fits operators to each learned edge function in isolation, making the discrete choice
  sensitive to initialisation"*; their **Greedy in-context SR** snaps edges by end-to-end loss
  improvement. → **Directly names our weakness:** `symbolic.py` fits each edge's φ in isolation
  (local R² argmax). Switch to greedy, output-loss-driven snapping.
- **pykan `auto_symbolic` / `suggest_symbolic`** — KindXiaoming/pykan docs (kan.html). **[FETCHED]**
  — `r2_threshold` (*"if r2 is below this threshold, the edge will not be fixed"*) + `weight_simple`
  (simplicity vs R²) + attribution-based `prune`. → **Lowest-effort upgrade:** add an
  R²-threshold + simplicity gate so we never force a formula onto an unfaithful edge.
- **KAN 2.0: KANs Meet Science (MultKAN)** — Liu et al., 2024. arXiv:2408.10205. **[FETCHED]** —
  multiplication nodes, kanpiler, tree converter. → MultKAN can natively express the
  **delay × current-state product** in the regenerative DDE that additive-spline gates cannot.
- **Softly Symbolifying KANs (S2KAN)** — Bagrow & Bongard, 2025. arXiv:2512.07875. **[FETCHED] ★**
  — differentiable, MDL-guided sparsification: symbolic when it suffices, *"gracefully degrades
  to dense splines"* otherwise. → train-time alternative to post-hoc snapping.
- **KAN-SR** — Bühler & Guillén-Gosálbez, 2026 (Comp. & Chem. Eng. 213). arXiv:2509.10089.
  **[FETCHED]** — divide-and-conquer + symmetry/separability simplification; pairs KAN symbolic
  extraction with neural CDEs for **dynamical systems** (closest peer-reviewed analog to our
  temporal/DDE setting).
- efficient-kan (Blealtan) **[FETCHED]** — fast B-spline kernels but **drops** the original L1
  sparsification (its sparsity ≠ our edge-importance signal). FastKAN (ZiyaoLi) **[FETCHED]** —
  RBF≈B-spline, 3.3× faster, **no symbolic extraction**. Original KAN (ICLR 2025, arXiv:2404.19756)
  **[FETCHED]** — anchor citation.

## B. Temporal / recurrent KANs (baselines to ablate against)

- **TKAN: Temporal Kolmogorov-Arnold Networks** — Genet & Inzirillo, 2024. arXiv:2405.07344;
  `tkan` PyPI 0.4.3 + github.com/remigenet/TKAN. **[FETCHED]** — RKAN layers + LSTM-style gating.
  → **Our literal baseline**; our novelty is the *explicit* τ=round(fs/tpf) delay vs TKAN's
  implicit gated memory. Headline ablation. (Keras-first; reimplement natively for a fair speed test.)
- **KAN-based GRU/LSTM for loan default** — Yang et al., 2025. arXiv:2507.13685. **[FETCHED]** —
  the closest published **KAN-GRU classifier**; mirror its baseline panel (GRU/LSTM/LSTM-Attn/
  LSTM-Transformer) and AUC/F1 metric set for our chatter head-to-head.
- **SigKAN** — Inzirillo & Genet, 2024. arXiv:2406.17890. **[FETCHED]** — learnable path-signature
  weighting (we already have a sigKAN stub off-by-default). **TKAT** (arXiv:2406.02486) **[FETCHED]**
  — attention sibling, no reported numbers.
- **KAN for TS Classification & Robustness** — Dong et al., 2024. arXiv:2408.07314. **[FETCHED]** —
  KAN ≈/slightly > MLP on 128 datasets, attributed to lower Lipschitz constant (→ noise-robustness
  ablation angle for our noisy datasets). **Exploring KANs for Interpretable TS Classification** —
  Barašin et al., 2025 (Int. J. Intelligent Systems, **peer-reviewed**). arXiv:2411.14904
  **[FETCHED]** — Efficient-KAN + SHAP; one of few peer-reviewed TS-classification KAN refs.
- **KAN vs LSTM in TS Forecasting** — Rather et al., 2025. arXiv:2511.18613. **[FETCHED]** —
  *"LSTM demonstrates substantial superiority"*; plain KAN loses → motivates **why hybrid + physics**.
  (Specific RMSE figures were search-only → **unverified, not cited.**)

## C. Chatter detection ML/DL — physics-informed & explainable

- **Explainable fault + severity classification with KANs** — Rigas et al., 2025 (Entropy;
  arXiv:2412.01322). **[FETCHED] ★** — KAN symbolic activations + feature attribution for joint
  fault **and severity**, *"100% F1 in most cases"* on bearing data. → **Closest precedent to our
  whole white-box approach** (KAN + symbolic + severity, like our b/blim head); related-work anchor
  that validates the method.
- **PINN milling-stability via deep subdomain adaptation (DSAN)** — Zhao, Jin, Wan, Hong, 2025
  (MSSP; SSRN 4944644). **[SEARCH-VERIFIED]** (ScienceDirect/SSRN 403) — physics-of-milling +
  domain-adaptation discrepancy loss to close sim→real. → its discrepancy loss is a concrete fix
  for our **clean↔noisy distribution shift / mirror-leakage** concern.
- **ML online stability-lobe-diagram + chatter suppression** — Huang et al., 2025 (ACC).
  arXiv:2511.17894. **[FETCHED]** — chatter *"modeled by delay differential equations"*; NN learns
  damping/natural-freq/cutting-coeffs (48.5% roughness reduction). → template for mapping TKAN
  latents to **named modal/cutting parameters**.
- **Interpretable anti-noise CNN (AFBAM) for thin-wall milling chatter** — Lu et al., 2024 (MSSP
  206:110885). **[SEARCH-VERIFIED]** (ScienceDirect 403) — interpretable frequency-band attention.
  → an AFBAM-style front-end could expose *which spectral bands* drive a chatter call (our weak
  attribution-spectrum result).
- **Multi-modal denoised hybrid NN** — Gao et al., 2025 (Sci. Rep. 15:3953). **[FETCHED]** —
  CEEMD-SVD denoise + dual-CNN/BiGRU/attention, **96.61% acc** → current SOTA baseline + drop-in
  denoiser for noisy data. **Small-sample GAN+attention** — Gao et al., 2025 (Sound & Vibration
  59(3)). **[FETCHED]** — WGAN augmentation, 84% acc at 24:1:1 imbalance → fixes our class imbalance.
- **DL chatter via FFT-image transfer (VGG16/ResNet50)** — Kounta et al., 2023. arXiv:2303.11430.
  **[FETCHED]** — 99% train → **73.7% under noise**; honest generalization-failure comparator
  (the overfitting/leakage trap we guard against).

## D. SINDy-for-DDE & time-series attribution (cross-check + better attribution)

- **Data-driven Discovery of DDEs with Discrete Delays** — Pecile, Demo, Tezzele, Rozza, Breda,
  2024. arXiv:2407.19640. **[FETCHED] ★** — augmented SINDy library + Bayesian-opt over τ;
  *"to identify a possibly unknown delay we minimize the reconstruction error over a set of
  candidates."* → **independent white-box DDE to cross-check the TKAN's learned τ** (chatter's
  regenerative delay is a single discrete delay = the easy case).
- **SINDy for DDEs (bacterial zinc)** — Sandoz et al., 2022 (Proc. R. Soc. A). arXiv:2208.10826.
  **[FETCHED]** — bilevel: SINDy at fixed τ, then optimize τ. **Stochastic-DDE SINDy** — Breda et
  al., 2025. arXiv:2508.03040. **[FETCHED]** — handles process noise (our noisy split). **Review:
  Data-driven methods for DDEs** — Breda et al., 2025. arXiv:2512.04894. **[FETCHED]** — survey to
  position TKAN vs SINDy-DDE / Neural-DDE.
- **pysindy reality check** **[FETCHED]** — *"Delay differential equations are not natively
  supported by PySINDy"*; no Hankel/delay library. → must build delayed columns by hand
  (`CustomLibrary`/`GeneralizedLibrary`) or use the papers' research code.
- **HAVOK** — Brunton et al., 2017 (Nature Comms). arXiv:1608.05306. **[FETCHED]** — Hankel + SVD →
  linear delay-coordinate model with intermittent forcing; the forcing bursts are an interpretable
  **chatter-onset** signal to validate the TKAN against. **Structured time-delay models** (Hirsh et
  al., 2021, Proc. R. Soc. A, PMC8511787) **[FETCHED]**; **Deep Delay Autoencoders** (Bakarji et al.,
  2022, arXiv:2201.05136) **[FETCHED]** for partial-state (single-channel) recovery.
- **TimeSHAP** — Bento et al., 2021 (KDD). arXiv:2012.00073. **[FETCHED]** — feature/timestep/cell
  attributions for recurrent models, beyond IG.
- **Benchmarking TS Interpretability + Temporal Saliency Rescaling (TSR)** — Ismail et al., 2020
  (NeurIPS). arXiv:2010.13924. **[FETCHED]** — saliency on TS *"conflat[es] time and feature
  domains"*; TSR disentangles *which lag* from *which feature*.
- **Input-Cell Attention** — Ismail et al., 2019 (NeurIPS). arXiv:1910.12370. **[FETCHED]** —
  *"RNN saliency vanishes over time, biasing detection of salient features only to later time
  steps."* → **critical caveat:** our IG attribution may systematically **under-credit the lagged
  (delayed) timesteps** — exactly where the regenerative term lives.
- **Sanity Checks for Saliency Maps** — Adebayo et al., 2018 (NeurIPS). arXiv:1810.03292.
  **[FETCHED]** — model/label randomization tests. → **we already implement this** as
  `faithfulness.randomization_sanity` (cosine −0.01 on fold 2 = passes).

---

## How this maps to OUR pipeline (priority order)

1. **Trust the ablation over the gradient for the delay claim, and add TSR.** Input-Cell Attention
   + TSR explain why our *attribution-based* delay signals look modest (x_delay ≈10% weight share,
   0.6% tpf-band energy, lag-peak at the window's end): **RNN saliency vanishes for early/lagged
   steps and conflates time-vs-feature.** Our **intervention** results — delay-ablation (−0.435 AUC,
   MAE 0.16→11.6) and the τ-sweep peak at the physical τ — are not subject to that bias and are the
   trustworthy evidence. Add Temporal Saliency Rescaling to the attribution path; expect the delay's
   apparent attribution to rise.
2. **Upgrade `symbolic.py`** with pykan's R²-threshold + simplicity gate (easy) and, later,
   greedy end-to-end-loss snapping (2603.15250) — fixes the per-edge-isolation init-sensitivity.
3. **SINDy-DDE cross-check** (2407.19640): recover τ + a regenerative term directly from the cutting
   signals and compare to the TKAN's learned delay — a strong, independent white-box validation for
   the paper. (pysindy = regression backbone; build delayed columns ourselves.)
4. **Baselines/ablations**: vanilla TKAN (2405.07344) vs our delay-aware KAN-GRU; KAN-GRU-classifier
   metric panel from 2507.13685; HAVOK (1608.05306) as an independent delay-coordinate comparator.
5. **Robustness/data**: CEEMD-SVD denoiser + WGAN imbalance fix (Gao 2025); DSAN discrepancy loss
   (Zhao 2025) for clean↔noisy shift. Rigas 2412.01322 = related-work anchor (KAN+symbolic+severity).

**Unverified / omitted (do not cite without checking):** specific RMSE figures in 2511.18613
(search-only); a "91% symbolic-extraction error" once attributed to 2503.15209 (not in the abstract
on direct read); gated ScienceDirect/Nature/SSRN pages tagged [SEARCH-VERIFIED] above (metadata
confirmed, full text not read).
