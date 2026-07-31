#!/bin/bash
# Pull Phase 3 results back from Great Lakes. Run from /Users/yash/Downloads/PRI2M/SINDy_DDE/.
#
# Mirrors sync_phase3_tkan_to_hpc.sh's connection handling: ONE master SSH
# connection (single Duo prompt), every transfer multiplexed over it.
#
# Deliberately does NOT fetch checkpoints (*.pt). They are GB-scale and every
# analysis script that needs them runs on GL against the live copy.

set -euo pipefail

REMOTE=yashkulk@greatlakes.arc-ts.umich.edu
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
for sub in regenerative counterfactuals activation_modes \
           fold0 fold1 fold2_w512 fold3 fold4 conformal concept_probes; do
  if ssh -o ControlPath="$CTRL" "$REMOTE" "test -d '$DEST/results/phase3/xai/$sub'"; then
    rs "$REMOTE:$DEST/results/phase3/xai/$sub/" "results/phase3/xai/$sub/"
  else
    echo "    ($sub not on GL yet)"
  fi
done

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
