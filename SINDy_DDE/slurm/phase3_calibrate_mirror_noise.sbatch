#!/bin/bash
#SBATCH --job-name=pri2m-noise-cal
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase3_calibrate_mirror_noise_%j.log
#SBATCH --error=logs/phase3_calibrate_mirror_noise_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=END,FAIL

# Noise comparison: measure mirror - clean (D vs D+60 / D+90, D = 1-30) and
# write the rel_sigma_per_unit that the noise-injection passes need
# (scripts/score_files_phase3.py and the MiniROCKET runner). Run BEFORE
# slurm/phase3_score_files.sbatch.
#
#   sbatch slurm/phase3_calibrate_mirror_noise.sbatch
#   # quick look first: env MAX_FILES=20 sbatch --export=ALL slurm/phase3_calibrate_mirror_noise.sbatch

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
NPZ_ROOT="${NPZ_ROOT:-$PROJECT_ROOT/phase2_signals}"
DATASETS="${DATASETS:-1-30}"
MAX_FILES="${MAX_FILES:-}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/results/phase3/xai/noise_compare/mirror_noise_calibration.json}"
PY="${PY:-python}"

cd "$PROJECT_ROOT"
if [[ -z "${SKIP_ENV:-}" ]]; then
  module load python3.11-anaconda
  source /home/$USER/envs/prism/bin/activate
fi
mkdir -p logs "$(dirname "$OUTPUT")"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -d "$NPZ_ROOT" ]]; then echo "ERROR: missing NPZ root $NPZ_ROOT" >&2; exit 1; fi

MAX_FLAG=()
[[ -n "$MAX_FILES" ]] && MAX_FLAG=(--max-files-per-dataset "$MAX_FILES")

echo "=== PRI2M mirror-noise calibration ==="
echo "NPZ root: $NPZ_ROOT   datasets: $DATASETS   max files: ${MAX_FILES:-all}"
echo "Started:  $(date)"

"$PY" scripts/calibrate_mirror_noise.py --npz-root "$NPZ_ROOT" \
  --datasets "$DATASETS" ${MAX_FLAG[@]+"${MAX_FLAG[@]}"} --output "$OUTPUT"

python3 - "$OUTPUT" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m.get("complete") is True, "calibration not complete"
s = m["summary"]
print(f"pairs={m['n_pairs']} missing={m['n_missing']} meta_disagree={m['n_meta_disagree']}")
print(f"rel_sigma_per_unit={s['rel_sigma_per_unit']} per_tier={s['per_tier']}")
print(f"linear_ok={s['linear_ok']} frac_additive={s['frac_additive']} undecided={s['undecided_reason']}")
if s["rel_sigma_per_unit"] is None:
    # A refusal is a finding, not a crash: the real-data scoring passes still
    # run; only the val_noise pass refuses (with this reason) later.
    print("WARNING: calibration REFUSED -- the val_noise passes will refuse to run.",
          file=sys.stderr)
if m["n_meta_disagree"]:
    print(f"WARNING: {m['n_meta_disagree']} pairs disagree on omega/label -- pairing suspect.",
          file=sys.stderr)
PY
echo "Finished: $(date)"
