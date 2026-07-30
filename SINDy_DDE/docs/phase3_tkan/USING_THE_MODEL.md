# Using the PRI2M chatter model

A trained TKAN model that scores a milling signal for **chatter**. This page is
everything you need to run it. No training, no GPU, no setup beyond the project
environment.

## Quickstart

On Great Lakes:

```bash
cd /nfs/turbo/coe-jaabell/PRI2M
sbatch --export=ALL,DATASET=6,OUT=$HOME/preds_ds6.csv slurm/phase3_predict.sbatch
```

That is the whole thing. Everything else defaults: which checkpoint, where the
signals are, where the file table is.

**Submit it, don't run it on the login node.** Every dataset has more than a
hundred files, and the script refuses login-node runs above 50. If you are
already inside a job or an `salloc`, call it directly:

```bash
module load python3.11-anaconda
source /home/$USER/envs/prism/bin/activate
export PYTHONPATH=$PWD/src:$PYTHONPATH

python scripts/predict_phase3.py --dataset 6 --out ~/preds_ds6.csv
```

`DATASET` / `--dataset` takes `all`, one id, or a comma list (`6,15,16`).

**Two limits worth knowing before you pick a dataset:**

- **Datasets 1–120 only**, with the default table. The 121–155 held-out sets
  live in a separate features CSV — pass it with `--table` if you need them.
- **Memory scales with how much you ask for.** Every selected file's full signal
  is loaded into RAM up front. 32 GB covers one dataset; for `all`, shard by
  dataset and concatenate the CSVs.

Dataset 6 is the suggested first run because the default model never trained on
it (see `in_train_split` below), so the probabilities you get back are honest.

## What you get

`preds.csv`, one row per file:

| column | meaning |
|---|---|
| `file` | the signal file that was scored |
| `prob` | **chatter probability**, mean of `sigmoid(logit)` over that file's windows |
| `n_windows` | how many 1024-sample windows the file produced |
| `dataset` | dataset id |
| `label` | the table's stored ground truth, for comparison — *not* a model input |
| `in_train_split` | **True means the model was trained on this file** — see below |

Plus `preds.csv.provenance.json`: exactly which weights ran (sha256), which code
commit, and how many rows were flagged. Attach it to anything you report.

Threshold `prob > 0.5` for a yes/no call. That threshold is where the published
accuracy below was measured.

## How good is it

Fold 4 on the strict held-out set (datasets 122–125, 855 files never seen in any
fold):

| metric | value |
|---|---|
| AUC | **0.920** |
| accuracy @ 0.5 | **0.841** |

Cite those numbers.

## Four things that will bite you

1. **Do not cite 0.955.** You may see that cross-validation AUC in older notes
   and slides. It is inflated by twin-dataset leakage — datasets 126–155
   replicate the operating points of 1–30. The honest number is 0.920.

2. **Drop `in_train_split == True` rows before averaging anything.** The model
   memorized those files; including them makes any aggregate look better than
   it is. The script warns you on stderr when it flags rows.

   ```python
   df = pd.read_csv("preds.csv")
   clean = df[~df["in_train_split"]]
   ```

3. **There is no severity output, on purpose.** The model has a severity head
   but it is unreliable (a few files dominate its error, RMSE ≈ 2× MAE). It is
   not exposed. If you need severity, talk to me first.

4. **Trained on datasets 1–120 only.** Simulated milling, one parameter family.
   Transfer to a different machine, tool, or material is unmeasured — treat
   anything outside that range as an experiment, not a measurement.

## Options you might want

| flag | why |
|---|---|
| `--fold 0..4` | Score with a different fold to check variance. **Default (4) is what we report** — don't cite another fold without saying so. |
| `--dataset 12,13` | Restrict to specific datasets. |
| `--device cuda` | If you happen to have a GPU allocation. CPU is fine and is the default in the sbatch. |
| `--allow-train-overlap` | Silence the overlap warning when you *intend* to score training data. |
| `--zoo <path>` | Point at a different published version. |

## In a notebook

```python
from pri2m.phase3.zoo import resolve_zoo, resolve_entry, verify_entry
from pri2m.phase3.predict import predict_files

entry = resolve_entry(resolve_zoo())
verify_entry(entry)
preds = predict_files(model, eval_table, npz_root, preprocessing)
```

Load the model with `scripts.xai_phase3._load_model(str(entry.checkpoint), args, pre)` —
the CLI is the reference for how those pieces fit together.

## Where the model lives

```
/nfs/turbo/coe-jaabell/PRI2M/models/phase3_tkan/v1/
```

Read-only and versioned. `MODEL_CARD.md` and `manifest.json` in that directory
record each fold's run id, sha256, held-out metrics, and which parent groups it
trained on. The CLI verifies the sha256 before every run and refuses to load
weights that have changed.

Published versions are immutable. A new model is `v2`, never an edit to `v1`, so
a number you cited last month still reproduces.

## When something breaks

- `no manifest.json under ...` — you are not on Great Lakes, or turbo is not
  mounted. Set `$PRI2M_MODEL_ZOO` if you have a copy elsewhere.
- `sha256 mismatch` — stop and tell me. Do not use the result.
- `no files to score` — that dataset's NPZ signals are not extracted yet.

Questions, or you need something the CLI does not do: ask me.
