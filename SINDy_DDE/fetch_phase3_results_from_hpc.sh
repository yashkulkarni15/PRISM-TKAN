#!/bin/bash
# Pull Phase 3 results back from Great Lakes. Run from /Users/yash/Downloads/PRI2M/SINDy_DDE/.
#
# Mirrors sync_phase3_tkan_to_hpc.sh's connection handling: ONE master SSH
# connection (single Duo prompt), every transfer multiplexed over it.
#
# Deliberately does NOT fetch checkpoints (*.pt). They are GB-scale and every
# analysis script that needs them runs on GL against the live copy.

set -euo pipefail

# Pin a login node when the round-robin lands on a loaded one (it authenticates
# and then refuses the session):  REMOTE=yashkulk@gl-login2.arc-ts.umich.edu $0
REMOTE="${REMOTE:-yashkulk@greatlakes.arc-ts.umich.edu}"
DEST=/nfs/turbo/coe-jaabell/PRI2M           # live GL project root
CTRL="$HOME/.ssh/cm-pri2m-fetch.sock"

cleanup() { ssh -o ControlPath="$CTRL" -O exit "$REMOTE" 2>/dev/null || true; }
trap cleanup EXIT

echo ">>> Opening master SSH connection to Great Lakes."
echo ">>> Approve the Duo 2FA prompt ONCE; all transfers reuse this connection."
ssh -fN -M -o ControlPath="$CTRL" -o ControlPersist=15m "$REMOTE"
echo ">>> Master connection established."

RSH="ssh -o ControlPath=$CTRL"
rs() { rsync -av --exclude=__pycache__ --exclude='*.pyc' -e "$RSH" "$@"; }

mkdir -p results/phase3/xai

# --------------------------------------------------------------------------- #
# 1. Fuzz / physics-floor stressor results (2026-07-25 rerun).
#    Supersedes the local 2026-07-17 copies, which predate the band-notch
#    contrast and carry no "contrasts" key at all.
# --------------------------------------------------------------------------- #
echo; echo ">>> [1/3] fuzz manifests + figures (5 folds)"
rs "$REMOTE:$DEST/results/phase3/xai/fuzz/" results/phase3/xai/fuzz/

# --------------------------------------------------------------------------- #
# 2. Run summaries only -- quantile-severity retrain AND the --no-side-input
#    ablation that landed in results/phase3/runs. Piped through tar rather than
#    rsync --include/--exclude, which is unreliable under macOS openrsync.
# --------------------------------------------------------------------------- #
echo; echo ">>> [2/3] summary.json / preprocessing.json (no checkpoints)"
ssh -o ControlPath="$CTRL" "$REMOTE" \
  "cd '$DEST' && find results/phase3/runs results/phase3/runs_quantile \
     results/phase3/runs_determinism results/phase3/runs_seeds \
     -maxdepth 2 \( -name summary.json -o -name preprocessing.json \) \
     -print0 2>/dev/null | tar cf - --null -T -" \
  | tar xf - -C .

# --------------------------------------------------------------------------- #
# 3. U12-B TSR construction-artifact control, if it has run by now.
# --------------------------------------------------------------------------- #
echo; echo ">>> [3/3] tsr_control (skipped if not yet run)"
if ssh -o ControlPath="$CTRL" "$REMOTE" "test -d '$DEST/results/phase3/xai/tsr_control'"; then
  rs "$REMOTE:$DEST/results/phase3/xai/tsr_control/" results/phase3/xai/tsr_control/
else
  echo "    (not on GL yet -- submit slurm/phase3_tsr_control.sbatch first)"
fi

# --------------------------------------------------------------------------- #
# 4. U13: the regenerative spline test + the counterfactual manifest (which now
#    carries Monte Carlo survival and, when run with DICE=1, the DiCE arm).
# --------------------------------------------------------------------------- #
echo; echo ">>> [4/4] U13 results (regenerative + counterfactuals)"
# The whitebox fold manifests are the source for the cross-fold edge overlap,
# the randomization-sanity number and several update figures. They were absent
# from this list until 2026-07-30, when a local `rm -rf SINDy_DDE` during a
# branch rebuild deleted them and nothing here could bring them back.
for sub in regenerative counterfactuals activation_modes comb \
           heldout_pruning symbolic_substitution mode_ablation comb_ablation \
           recurrence tkan_delay_heldout noise_compare equations \
           fold0 fold1 fold2_w512 fold3 fold4 conformal concept_probes; do
  if ssh -o ControlPath="$CTRL" "$REMOTE" "test -d '$DEST/results/phase3/xai/$sub'"; then
    rs "$REMOTE:$DEST/results/phase3/xai/$sub/" "results/phase3/xai/$sub/"
  else
    echo "    ($sub not on GL yet)"
  fi
done

# Exp 3 (2026-10-05): MiniROCKET manifests live outside xai/.
echo; echo ">>> MiniROCKET comparator (Exp 3)"
if ssh -o ControlPath="$CTRL" "$REMOTE" "test -d '$DEST/results/phase3/minirocket'"; then
  rs "$REMOTE:$DEST/results/phase3/minirocket/" results/phase3/minirocket/
else
  echo "    (minirocket not on GL yet)"
fi

# --------------------------------------------------------------------------- #
# Verification. The whole point of the fuzz refetch is the "contrasts" key;
# if it is still absent the transfer did not do what it was run to do.
# --------------------------------------------------------------------------- #
echo
echo "================================================================"
python3 - <<'PY'
import json, glob, os, time

print("fuzz manifests:")
for p in sorted(glob.glob("results/phase3/xai/fuzz/*/fuzz_manifest.json")):
    d = json.load(open(p))
    c = sorted(d.get("contrasts", {})) or "*** STILL ABSENT ***"
    print(f"  {p.split('/')[-2]:8s} {os.path.getsize(p)/1024:6.0f} KB  "
          f"{time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p)))}  contrasts={c}")

print("\nU13:")
for p in sorted(glob.glob("results/phase3/xai/regenerative/*/regenerative.json")):
    d = json.load(open(p))
    s = d.get("summary", d)
    print(f"  regenerative {p.split('/')[-2]:12s} "
          f"{time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p)))}")
# The determinism repeat: two runs of the SAME fold and seed with reproducible
# kernels requested. Identical histories mean the flags fixed it.
det = sorted(glob.glob("results/phase3/runs_determinism/*/summary.json"))
print(f"\nDeterminism repeat: {len(det)} run(s) fetched")
for p in det:
    d = json.load(open(p)); e = d.get("environment") or {}
    print(f"  {p.split('/')[-2][:10]} fold {d.get('fold')} epochs {len(d.get('history',{}).get('val',[]))}"
          f"  node={e.get('slurm_nodelist')} gpu={e.get('gpu_name')}"
          f"  det={e.get('deterministic_algorithms')} bench={e.get('cudnn_benchmark')}")
    if not e:
        print("    *** no environment block: this run predates the determinism code ***")
if len(det) == 2:
    a, b = [json.load(open(p)) for p in det]
    same_cfg = a.get("config") == b.get("config")
    identical = a.get("history") == b.get("history")
    print(f"  same config: {same_cfg}")
    print(f"  VERDICT: {'BIT-IDENTICAL' if identical else 'DIVERGED'}")
    if not identical:
        va = [x.get("auc") for x in a.get("history", {}).get("val", [])]
        vb = [x.get("auc") for x in b.get("history", {}).get("val", [])]
        gaps = [abs(x - y) for x, y in zip(va, vb) if x is not None and y is not None]
        if gaps:
            print(f"    largest per-epoch AUC gap: {max(gaps):.4f}")
elif det:
    print("  (need both runs before the comparison means anything)")

print("\nComb probe (predicted vs measured, per fold):")
for p in sorted(glob.glob("results/phase3/xai/comb/comb_probe_*.json")):
    d = json.load(open(p))
    s = d.get("summary", {})
    tr = (s.get("trained_peak_alignment") or {}).get("frac_within_0_1")
    un = (s.get("untrained_peak_alignment") or {}).get("frac_within_0_1")
    cr = s.get("median_corr_predicted_measured")
    # chance is 0.20: the +/-0.1 bands around 0,1,2,3 cover 20% of a 0-3 axis.
    print(f"  {p.split('/')[-1]:32s} trained {tr}  untrained {un}  corr(pred,meas) {cr}")
if not glob.glob("results/phase3/xai/comb/comb_probe_*.json"):
    print("  (no comb probe manifests fetched)")

print("\nU14-B activation modes:")
for p in sorted(glob.glob("results/phase3/xai/activation_modes/*/activation_modes_manifest.json")):
    d = json.load(open(p))
    prov, by_k = d.get("provenance", {}), d.get("by_k", {})
    done = prov.get("complete")
    print(f"  {p.split('/')[-2]:10s} complete={done}  k={sorted(by_k, key=int)}"
          f"  seeds={prov.get('n_untrained_seeds')}")
    if not done:
        print("    *** PARTIAL SWEEP -- a manifest is written after every k, so"
              " presence alone proves nothing ***")
    # Verdicts are PER STRATUM; there is no top-level "verdict" key.
    for st in sorted({s for k in by_k for s in by_k[k].get("strata", {})}):
        cells = []
        for k in sorted(by_k, key=int):
            v = ((by_k[k].get("strata", {}).get(st) or {}).get("verdict") or {})
            a = v.get("artifact_suspected")
            cells.append(f"k{k}=" + ("UNDEC" if a is None else "ARTIFACT" if a else "survives"))
        print(f"    {st:<22} " + "  ".join(cells))

for p in sorted(glob.glob("results/phase3/xai/counterfactuals/*manifest.json")):
    pooled = json.load(open(p)).get("pooled", {})
    surv, dice = pooled.get("survival"), pooled.get("dice")
    print(f"  counterfactuals agreement={pooled.get('agreement_rate')}")
    print(f"    survival={'ABSENT' if not surv else surv.get('frac_above_0_85')}"
          f"  dice={'ABSENT' if not dice else dice.get('error') or dice.get('found_rate')}")

# Both of these runners write a partial manifest after EVERY arm, with
# "complete": false, so the file existing proves nothing -- a wall-killed run
# leaves a readable curve that looks exactly like a finished one.
print("\nHeld-out pruning (ranked vs size-matched random control):")
for p in sorted(glob.glob("results/phase3/xai/heldout_pruning/*/heldout_pruning.json")):
    d = json.load(open(p)); prov = d.get("provenance", {})
    base = d.get("baseline_auc")
    print(f"  {p.split('/')[-2]:10s} complete={prov.get('complete')}  "
          f"baseline={'None' if base is None else format(base, '.4f')}  "
          f"edges={prov.get('n_edges_total')}  tol={prov.get('tol')}  "
          f"smallest_faithful_fraction={d.get('smallest_faithful_fraction')}")
    if not prov.get("complete"):
        print("    *** PARTIAL -- written after every fraction; NOT a finished run ***")
    for r in d.get("rows", []):
        rk, rd = r.get("ranked_auc"), r.get("random_auc")
        f = lambda v: "None" if v is None else format(v, ".4f")
        gap = "" if (rk is None or rd is None) else f"  ranked-random={rk - rd:+.4f}"
        print(f"    keep {r.get('fraction'):<6} n_zeroed={str(r.get('n_zeroed')):>6}  "
              f"ranked={f(rk)}  random={f(rd)}{gap}"
              + (f"  [{r.get('error') or r.get('note')}]"
                 if r.get("error") or r.get("note") else ""))
if not glob.glob("results/phase3/xai/heldout_pruning/*/heldout_pruning.json"):
    print("  (nothing fetched)")

print("\nSymbolic substitution (fitted closed forms swapped in at forward time):")
for p in sorted(glob.glob("results/phase3/xai/symbolic_substitution/*/symbolic_substitution.json")):
    d = json.load(open(p)); prov = d.get("provenance", {})
    base = d.get("baseline_auc")
    print(f"  {p.split('/')[-2]:10s} complete={prov.get('complete')}  "
          f"baseline={'None' if base is None else format(base, '.4f')}  "
          f"k_sweep={prov.get('k_sweep')}  rejected_fits={prov.get('n_rejected_fits')}  "
          f"tol={prov.get('tol')}")
    if not prov.get("complete"):
        print("    *** PARTIAL -- written after every arm; NOT a finished run ***")
    for r in d.get("rows", []):
        c, q = r.get("correct_auc"), r.get("permuted_auc")
        f = lambda v: "None" if v is None else format(v, ".4f")
        v = r.get("verdict") or {}
        tag = f"k={r.get('k')}" + (" CLAMPED" if r.get("clamped") else "")
        print(f"    {tag:<12} n={str(r.get('n_substituted')):>3}  correct={f(c)}  "
              f"permuted={f(q)}  "
              + (f"[{r.get('error')}]" if r.get("error")
                 else f"PASSED={v.get('passed')} "
                      f"abs_gap={v.get('abs_gap')} control_gap={v.get('control_gap')}"))
if not glob.glob("results/phase3/xai/symbolic_substitution/*/symbolic_substitution.json"):
    print("  (nothing fetched)")

# Unlike the two above, this one writes its manifest ONCE at the end, so the
# file existing does mean the sweep finished -- but check the flag anyway.
print("\nU14-B step 8 mode ablation (a mode must beat BOTH controls by > floor):")
for p in sorted(glob.glob("results/phase3/xai/mode_ablation/mode_ablation_*.json")):
    d = json.load(open(p))
    prov, base = d.get("provenance", {}), d.get("baseline_auc")
    print(f"  {p.split('/')[-1]:28s} complete={d.get('complete')}  "
          f"baseline={'None' if base is None else format(base, '.4f')}  "
          f"k={prov.get('k')}  floor={prov.get('floor')}")
    for m in d.get("modes", []):
        v = m.get("verdict") or {}
        print(f"    {m.get('stratum'):<22} mode {m.get('mode')}  n={m.get('n_edges'):<4} "
              f"drop={v.get('mode_drop')}  random={v.get('random_matched_drop')}  "
              f"other={v.get('other_modes_drop')}  CLAIMABLE={v.get('claimable')}")
    # Exp 1b arms, present only in --pairs runs.
    for q in d.get("pairs", []) + d.get("all_gated", []):
        v = q.get("verdict") or {}
        drop = v.get("pair_drop", v.get("all_drop"))
        print(f"    PAIR/ALL modes {q.get('modes')}  n={q.get('n_edges'):<4} drop={drop}  "
              f"random={v.get('random_matched_drop')}  interaction={v.get('interaction')} "
              f"({v.get('interaction_class')})  CLAIMABLE={v.get('claimable')}")
    rc = d.get("reference_check")
    if rc is not None:
        print(f"    reference_check modes_identical={rc.get('modes_identical')}  "
              "(False/None means the clustering did NOT reproduce Exp 1 -- read before using)")
if not glob.glob("results/phase3/xai/mode_ablation/mode_ablation_*.json"):
    print("  (nothing fetched)")

print("\nExp 2 recurrence (compared by FUNCTION; None = UNDECIDED, not a fail):")
for p in sorted(glob.glob("results/phase3/xai/recurrence/recurrence_*.json")):
    d = json.load(open(p))
    print(f"  {p.split('/')[-1]}  complete={d.get('complete')}  "
          f"exclude={d.get('provenance', {}).get('exclude_mode')}")
    for name, n in (d.get("networks") or {}).items():
        print(f"    {name:6s} seed={n.get('train_seed')} claimable={n.get('claimable_modes')} "
              f"excluded={n.get('excluded_modes')} n_cluster={n.get('n_cluster_edges')}")
    for fold, s in (d.get("per_fold") or {}).items():
        print(f"    fold {fold}: {s.get('summary')}")
if not glob.glob("results/phase3/xai/recurrence/recurrence_*.json"):
    print("  (nothing fetched)")

print("\nExp 3 MiniROCKET (file-level AUC; drops must exceed the floor):")
for p in sorted(glob.glob("results/phase3/minirocket/minirocket_*.json")):
    d = json.load(open(p))
    print(f"  {p.split('/')[-1]}  complete={d.get('complete')}  "
          f"weighted={d.get('provenance', {}).get('sample_weighted')}")
    print(f"    {json.dumps({k: v for k, v in d.items() if k not in ('provenance',)}, default=str)[:600]}")
if not glob.glob("results/phase3/minirocket/minirocket_*.json"):
    print("  (nothing fetched)")

print("\nExp 3 matched TKAN delay test (held-out 121-125):")
for p in sorted(glob.glob("results/phase3/xai/tkan_delay_heldout/tkan_delay_heldout_*.json")):
    d = json.load(open(p))
    print(f"  {p.split('/')[-1]}  complete={d.get('complete')}  baseline={d.get('baseline_auc')}")
    print(f"    {json.dumps({k: v for k, v in d.items() if k not in ('provenance',)}, default=str)[:400]}")
if not glob.glob("results/phase3/xai/tkan_delay_heldout/tkan_delay_heldout_*.json"):
    print("  (nothing fetched)")

print("\nComb ablation (deleting the comb-carrying delay pairs):")
for p in sorted(glob.glob("results/phase3/xai/comb_ablation/comb_ablation_*.json")):
    d = json.load(open(p)); prov = d.get("provenance", {}); v = d.get("verdict") or {}
    print(f"  {p.split('/')[-1]:34s} complete={prov.get('complete')}  "
          f"gate={prov.get('corr_threshold')}  comb_edges={prov.get('n_comb_edges')}  "
          f"below_gate={prov.get('n_below_gate_edges')}")
    if not prov.get("complete"):
        print("    *** PARTIAL or abandoned -- NOT a finished run ***")
        if prov.get("abandoned"):
            print(f"    abandoned: {prov['abandoned']}")
    for name, why in (prov.get("control_errors") or {}).items():
        # An absent control is why claimable comes back None. It is not a pass.
        print(f"    CONTROL ABSENT {name}: {why}")
    for name, a in (d.get("arms") or {}).items():
        auc = a.get("auc")
        print(f"    {name:<26} n={a.get('n_edges'):>4} "
              f"imp={a.get('summed_importance'):8.2f} "
              f"auc={'None' if auc is None else format(auc, '.4f')}"
              + (f"  [{a['error']}]" if a.get("error") else ""))
    print(f"    drop={v.get('mode_drop')}  vs random={v.get('random_matched_drop')}  "
          f"vs nonproportional={v.get('other_modes_drop')}  "
          f"CLAIMABLE={v.get('claimable')}"
          + ("  (None = UNDECIDED, NOT a pass)" if v.get("claimable") is None else ""))
if not glob.glob("results/phase3/xai/comb_ablation/comb_ablation_*.json"):
    print("  (nothing fetched)")

print("\nrun summaries:")
for p in sorted(glob.glob("results/phase3/runs*/*/summary.json"), key=os.path.getmtime):
    d = json.load(open(p))
    cfg = d.get("config", {})
    v = d.get("best_auc_val", {})
    print(f"  {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p)))} "
          f"{p.split('/')[-3]:14s} {p.split('/')[-2][:10]} fold {d.get('fold')} "
          f"side={cfg.get('side_input')} sev={cfg.get('severity_mode')} "
          f"auc={v.get('auc', float('nan')):.4f} mae={v.get('mae', float('nan')):.3f}")
PY
echo "================================================================"
