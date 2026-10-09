#!/bin/bash
# What is the queue doing, and did the jobs that left it actually finish?
# Run from /Users/yash/Downloads/PRI2M/SINDy_DDE/.
#
# Connection handling mirrors sync_phase3_tkan_to_hpc.sh: ONE master SSH
# connection (single Duo prompt), every query multiplexed over it.
#
# Four sections, in the order you need them:
#   1. squeue   -- what is still queued or running, and WHY it is queued
#   2. sacct    -- what left the queue in the window, with exit codes
#   3. logs     -- automatic tail of .err for anything that did not exit clean
#   4. manifests-- the "complete" flag on disk
#
# Section 4 exists because of the standing rule this project keeps relearning:
# exit 0 is not evidence. The pruning, substitution and comb-ablation runners
# checkpoint a manifest after EVERY arm with "complete": false, so a wall-killed
# run leaves a readable file that looks exactly like a finished one. sacct says
# COMPLETED; the flag says whether the sweep got to the end.
#
# Usage:
#   ./check_hpc_jobs.sh              # last 3 days
#   DAYS=7 ./check_hpc_jobs.sh       # wider sacct window
#   JOBS="57299173 57299189" ./check_hpc_jobs.sh   # tail these logs regardless
#   REMOTE=yashkulk@gl-login2.arc-ts.umich.edu ./check_hpc_jobs.sh

set -euo pipefail

REMOTE="${REMOTE:-yashkulk@greatlakes.arc-ts.umich.edu}"
DEST="${DEST:-/nfs/turbo/coe-jaabell/PRI2M}"   # live GL project root, and the
                                               # submit dir, so logs/ is under it
DAYS="${DAYS:-3}"
JOBS="${JOBS:-}"                               # extra job ids to tail, optional
LINES="${LINES:-25}"                           # tail length per failed job
CTRL="$HOME/.ssh/cm-pri2m-${REMOTE##*@}.sock"

cleanup() { ssh -o ControlPath="$CTRL" -O exit "$REMOTE" 2>/dev/null || true; }
trap cleanup EXIT

echo ">>> Opening master SSH connection to Great Lakes."
echo ">>> Approve the Duo 2FA prompt ONCE; every query reuses this connection."
ssh -fN -M -o ControlPath="$CTRL" -o ControlPersist=15m "$REMOTE"
echo ">>> Connected."

# --------------------------------------------------------------------------- #
# Sections 1-3 run as one remote bash script. No `while read` loops in here:
# bash is reading this script from stdin, and a read loop would eat it.
# --------------------------------------------------------------------------- #
ssh -o ControlPath="$CTRL" "$REMOTE" bash -s -- "$DEST" "$DAYS" "$LINES" "$JOBS" <<'REMOTE'
set -uo pipefail
DEST="$1"; DAYS="$2"; LINES="$3"; EXTRA_JOBS="${4:-}"
SINCE="now-${DAYS}days"

echo
echo "================================================================"
echo "1. IN THE QUEUE  (squeue -u $USER)"
echo "================================================================"
# REASON is the column that matters when nothing is starting: AssocGrpGRES is
# the account-wide GPU cap, Priority/Resources is just the queue, and
# QOSMaxWallDurationPerJobLimit means the walltime was rejected outright.
squeue -u "$USER" -o '%.12i %.18j %.10P %.9T %.11M %.11l %.6D %.20R' -S T,-M
n=$(squeue -u "$USER" -h | wc -l)
echo "-- $n job(s) in queue"

echo
echo "================================================================"
echo "2. LEFT THE QUEUE SINCE $SINCE  (sacct)"
echo "================================================================"
# -X = allocations only, no .batch/.extern step rows. ExitCode 0:0 on a
# COMPLETED row is the only clean outcome; TIMEOUT still says 0:0 sometimes.
sacct -u "$USER" -S "$SINCE" -X \
      --format=JobID%12,JobName%22,Partition%10,State%18,ExitCode%8,Elapsed%11,Timelimit%11,Start%16,End%16

echo
echo "================================================================"
echo "3. LOGS FOR ANYTHING THAT DID NOT EXIT CLEAN"
echo "================================================================"
# Everything that is not exactly "COMPLETED". Catches FAILED, TIMEOUT,
# OUT_OF_MEMORY, NODE_FAIL, CANCELLED -- the last of which is what a job looks
# like when it hits the wall on some paths.
BAD=$(sacct -u "$USER" -S "$SINCE" -X -n -P --format=JobID,State \
      | awk -F'|' '$2 != "COMPLETED" {split($1,a,"."); print a[1]}' | sort -u)
BAD="$BAD $EXTRA_JOBS"
BAD=$(echo $BAD | tr ' ' '\n' | grep -v '^$' | sort -u || true)

if [ -z "$BAD" ]; then
  echo "  (nothing failed, timed out or was cancelled in the window)"
else
  for jid in $BAD; do
    state=$(sacct -j "$jid" -X -n -P --format=State | head -1)
    echo
    echo "---- job $jid  [$state] ----------------------------------------"
    # Per-step rows carry MaxRSS and the real exit code; the allocation row does
    # not. An OOM shows up here and nowhere else.
    sacct -j "$jid" --format=JobID%14,JobName%16,State%16,ExitCode%8,MaxRSS%10,Elapsed%10
    for f in "$DEST"/logs/*_"$jid".err "$DEST"/logs/*_"$jid".log; do
      [ -f "$f" ] || continue
      sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
      echo "  --- $f  (${sz} bytes) ---"
      [ "$sz" -gt 0 ] && tail -n "$LINES" "$f" | sed 's/^/    /'
    done
  done
fi

echo
echo "================================================================"
echo "3b. NEWEST LOGS ON DISK (sanity: are they growing?)"
echo "================================================================"
ls -lt --time-style='+%m-%d %H:%M' "$DEST"/logs/ 2>/dev/null | head -12 || echo "  (no logs dir)"
REMOTE

# --------------------------------------------------------------------------- #
# 4. Completeness on disk. Separate ssh call so python reads its own script off
#    a clean stdin rather than fighting the bash heredoc above.
# --------------------------------------------------------------------------- #
echo
echo "================================================================"
echo "4. MANIFESTS ON DISK -- 'complete' flag, not exit code"
echo "================================================================"
ssh -o ControlPath="$CTRL" "$REMOTE" "cd '$DEST' && python3 -" <<'PY'
import glob, json, os, time

# The flag lives at top level on some runners and under provenance on others.
def completeness(d):
    for key in ("complete",):
        if key in d:
            return d[key]
    return (d.get("provenance") or {}).get("complete")

PATTERNS = [
    "results/phase3/xai/*/*.json",
    "results/phase3/xai/*/*/*.json",
    "results/phase3/runs*/*/summary.json",
]
seen, rows = set(), []
for pat in PATTERNS:
    for p in glob.glob(pat):
        if p in seen or os.path.basename(p) == "preprocessing.json":
            continue
        seen.add(p)
        try:
            d = json.load(open(p))
        except Exception as e:
            rows.append((os.path.getmtime(p), p, f"UNREADABLE ({e.__class__.__name__})", 0))
            continue
        if not isinstance(d, dict):
            continue
        c = completeness(d)
        # summary.json has no flag; its stand-in is that training history exists.
        if c is None and "history" in d:
            c = bool(d.get("best_auc_val"))
        if c is None:
            continue
        rows.append((os.path.getmtime(p), p, c, os.path.getsize(p)))

rows.sort(reverse=True)
cut = time.time() - 14 * 86400
for mt, p, c, sz in rows[:40]:
    age = "" if mt > cut else "  (>14d old)"
    tag = {True: "COMPLETE ", False: "*** PARTIAL ***"}.get(c, str(c))
    print(f"  {time.strftime('%m-%d %H:%M', time.localtime(mt))}  {tag:<16} "
          f"{sz/1024:7.0f} KB  {p}{age}")
if not rows:
    print("  (no manifests with a completeness flag found)")
print()
print("  PARTIAL = the runner checkpointed mid-sweep and never reached the end.")
print("  Read nothing off a partial manifest; it looks exactly like a finished one.")
PY
echo "================================================================"
