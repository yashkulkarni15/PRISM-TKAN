#!/bin/bash
#SBATCH --job-name=pri2m-p3-tsr-merge
#SBATCH --account=jaabell0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/phase3_tsr_merge_%j.log
#SBATCH --error=logs/phase3_tsr_merge_%j.err
#SBATCH --mail-user=yashkulk@umich.edu
#SBATCH --mail-type=FAIL

# Fold the sharded TSR results (results/phase3/xai/<TAG>/tsr_shards/*.json) back
# into the fixed-stage manifest (whitebox_manifest.json from the --skip-tsr run),
# reproducing the exact results["tsr"] block a single un-sharded run would write
# and refreshing the render-ready report. Refuses to run on an incomplete shard
# set, so submit this with --dependency=afterok:<ARRAY_JOBID> so it only fires
# once every array task has succeeded (run_whitebox_sharded.sh wires this).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/nfs/turbo/coe-jaabell/PRI2M}"
TAG="${TAG:-fold2}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/phase3/xai/$TAG}"

cd "$PROJECT_ROOT"

module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== PRI2M Phase 3 white-box TSR merge ==="
echo "Tag:    $TAG"
echo "Output: $OUTPUT_DIR"
echo "Started: $(date)"

python scripts/merge_tsr_shards.py --output-dir "$OUTPUT_DIR"

echo "Finished: $(date)"
echo "Render the report locally after rsync:"
echo "  python scripts/render_whitebox_report.py --manifest $OUTPUT_DIR/whitebox_manifest.json"
