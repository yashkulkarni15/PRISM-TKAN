"""Phase 3 training harness (Task 3).

A model-agnostic trainer that wires the Phase 3 building blocks together:

* :func:`build_model` constructs a :class:`~pri2m.models.heads.Phase3Model`
  (``TKANModel`` for ``"tkan"``/``"sigkan"`` or
  :class:`~pri2m.models.baseline.MLPBaseline` for ``"mlp"``) from a
  :class:`TrainConfig`.
* :class:`Phase3Trainer` runs the Adam optimization loop over the noise-weighted
  :func:`~pri2m.training.losses.total_loss`, evaluates per-file severity metrics
  on the validation loader, checkpoints each epoch (plus ``best.pt`` / ``final.pt``),
  and runs the trajectory stage-gate.
* :func:`train_one_fold` is the orchestrator: build leakage-safe fold loaders,
  build the model, open an MLflow run (``file:`` URI), fit, log params/metrics,
  run the gate, write ``summary.json``, prune periodic checkpoints, and return
  the summary dict.

Determinism is pinned by seeding ``random`` / ``numpy`` / ``torch`` at the start
of :meth:`Phase3Trainer.fit` and :func:`train_one_fold`. MLflow autologging is
left off; a ``file:`` tracking URI keeps everything local and quota-safe.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import numpy as np
import torch

# MLflow 3.x put the filesystem tracking backend in maintenance mode and raises
# unless this opt-out is set. The Phase 3 design mandates a ``file:`` tracking
# URI (quota-safe, no server on Great Lakes), so we opt in here unless the
# operator has already chosen a value. Set before any ``import mlflow``.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from pri2m.config import load_config
from pri2m.data.phase3_sequence import NormStats, WindowConfig, build_fold_loaders
from pri2m.models.baseline import MLPBaseline
from pri2m.models.heads import _VALID_SEVERITY_MODES, Phase3Model
from pri2m.models.tkan_model import DEFAULT_SIDE_FEATURES, TKANModel
from pri2m.training.losses import LossWeights, total_loss
from pri2m.training.metrics import per_file_severity_metrics, trajectory_gate_report
from pri2m.training.mlflow_retention import apply_checkpoint_retention

__all__ = [
    "TrainConfig",
    "build_model",
    "build_optimizer",
    "schedule_factor",
    "side_feature_stats",
    "Phase3Trainer",
    "train_one_fold",
]

_TKAN_MODELS = {"tkan", "sigkan"}
_SMOKE_MAX_EPOCHS = 2
# Smoke runs cap the window so the default 1024-length geometry still fits the
# short synthetic signals used in CI without the caller hand-tuning it.
_SMOKE_MAX_WINDOW = 64


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Hyper-parameters + run options for a single Phase 3 fold.

    ``model`` selects the architecture (``"tkan"``/``"sigkan"`` -> ``TKANModel``,
    ``"mlp"`` -> ``MLPBaseline``). ``smoke`` clamps the epoch count to
    :data:`_SMOKE_MAX_EPOCHS` for fast CI runs.
    """

    model: str = "tkan"  # {"tkan","mlp","sigkan"}
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 0
    window: WindowConfig = field(default_factory=WindowConfig)
    hidden_size: int = 64
    num_layers: int = 1
    grid_size: int = 5
    spline_order: int = 3
    loss: LossWeights = field(default_factory=LossWeights)
    fold: int = 0
    with_trajectory: bool = False
    traj_steps: int = 16
    # Temporal average-pool factor for the TKAN trunk (1 == no pooling). Pooling
    # shortens the recurrence p-fold; ignored by the MLP baseline.
    pool: int = 1
    smoke: bool = False
    # ``"auto"`` -> CUDA when available else CPU; or an explicit "cuda"/"cpu".
    device: str = "auto"
    # DataLoader worker processes (0 == single-process, the safe default).
    num_workers: int = 0
    # torch.compile the recurrent cells (CUDA-only, metric-safe ~3.4x on A40).
    compile: bool = False
    # -- optimizer stability bundle (fold-0 instability fix; all flag-gated,
    # defaults reproduce the original fixed-lr Adam behavior exactly) -------- #
    # Per-epoch lr schedule applied via LambdaLR: {"constant","cosine"}.
    lr_schedule: str = "constant"
    # Linear lr warmup over this many leading epochs (0 == off).
    warmup_epochs: int = 0
    # clip_grad_norm_ max-norm between backward and step; <= 0 means off.
    grad_clip: float = 0.0
    # Optimizer family: {"adam","adamw"} (AdamW == decoupled weight decay).
    optimizer: str = "adam"
    # Weight decay passed to the optimizer (0.0 == off, the Adam default).
    weight_decay: float = 0.0
    # Gao-style process-parameter side branch (omega/tpf/tau concatenated onto
    # the encoder context; tkan/sigkan only). Off by default so the existing
    # runs and the MLP baseline are untouched.
    side_input: bool = False
    # Width of the side-branch MLP output (ignored unless side_input is True).
    side_hidden: int = 16
    # Severity-head regression mode: {"huber","quantile"} (tkan/sigkan only).
    # "huber" is the validated point head; "quantile" emits one column per level
    # in ``severity_quantiles`` and routes the regression term through the
    # pinball loss (see pri2m.training.losses.severity_loss).
    severity_mode: str = "huber"
    # Quantile levels for severity_mode="quantile" (ignored in "huber" mode).
    severity_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def effective_epochs(self) -> int:
        """Epoch count after the smoke clamp."""
        if self.smoke:
            return min(self.epochs, _SMOKE_MAX_EPOCHS)
        return self.epochs

    def effective_window(self) -> WindowConfig:
        """Window geometry after the smoke clamp.

        Under ``smoke`` the window length/stride are capped at
        :data:`_SMOKE_MAX_WINDOW` so the default 1024-length geometry still
        produces windows on the short synthetic signals used in CI. Channels are
        preserved. Non-smoke runs return the configured window unchanged.
        """
        if not self.smoke:
            return self.window
        length = min(self.window.length, _SMOKE_MAX_WINDOW)
        stride = min(self.window.stride, length)
        return WindowConfig(length=length, stride=stride, channels=self.window.channels)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def _seed_everything(seed: int) -> None:
    """Seed ``random`` / ``numpy`` / ``torch`` for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU CI
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Device handling
# --------------------------------------------------------------------------- #
def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec to a ``torch.device``.

    ``"auto"`` (or empty) selects CUDA when available, otherwise CPU. Any other
    value (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``) is passed through verbatim.
    """
    if spec and str(spec).lower() not in ("auto", ""):
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strip_compile_prefix(state_dict: dict) -> dict:
    """Drop the ``_orig_mod.`` segments ``torch.compile`` inserts into keys.

    Compiling a submodule (see :func:`maybe_compile_cells`) wraps it in an
    ``OptimizedModule`` whose ``state_dict`` keys gain an ``_orig_mod.`` segment.
    Stripping it keeps every checkpoint loadable by a plain (uncompiled)
    ``TKANModel`` -- the XAI / inference path builds an uncompiled model and
    ``load_state_dict`` matches keys exactly. A no-op on already-clean keys.
    """
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def maybe_compile_cells(model, cfg: "TrainConfig", device: torch.device):
    """``torch.compile`` the recurrent TKAN cells in place when enabled on CUDA.

    The KAN-GRU recurrence is GPU-kernel-launch-bound; compiling the *cell*
    (stable shapes) fuses its elementwise/spline storm into a few kernels --
    measured ~3.4x faster per batch on an A40. No-op unless ``cfg.compile`` is
    set AND we are on CUDA (CPU inductor is far too slow for this recurrence and
    there is no launch overhead to recover on CPU). The speedup is pure kernel
    fusion of the launch-bound recurrence -- numerically near-identical (fp32,
    ULP-level reassociation only). TF32 is intentionally NOT enabled: it drops
    matmul precision and perturbs this precision-sensitive (spline) training. The
    model is mutated in place and returned.
    """
    if not getattr(cfg, "compile", False):
        return model
    if device.type != "cuda" or not torch.cuda.is_available():
        return model
    layers = getattr(getattr(model, "encoder", None), "layers", None)
    if layers is None:
        return model
    for i in range(len(layers)):
        layers[i] = torch.compile(layers[i])
    return model


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move a collated batch's tensor values to ``device`` (lists pass through).

    A no-op on CPU so the single-process CPU path stays allocation-free.
    """
    if device.type == "cpu":
        return batch
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(cfg: TrainConfig) -> Phase3Model:
    """Build the concrete Phase 3 model named by ``cfg.model``.

    ``"tkan"`` and ``"sigkan"`` both map to :class:`~pri2m.models.tkan_model.TKANModel`
    (the signature-weighting trunk is an optional enhancement, not required here).
    ``"mlp"`` maps to :class:`~pri2m.models.baseline.MLPBaseline`.

    The MLP's input width tracks :meth:`TrainConfig.effective_window` so it stays
    consistent with the windows the loaders actually emit under ``smoke``.

    ``cfg.side_input`` enables the Gao-style process-parameter side branch on
    the TKAN models only; requesting it for ``"mlp"`` raises ``ValueError`` so
    the MLP baseline comparison stays untouched.

    ``cfg.severity_mode`` selects the severity head: ``"huber"`` (default) is
    the validated point regression, ``"quantile"`` emits one column per level in
    ``cfg.severity_quantiles`` and is trained with the pinball loss. Like
    ``side_input`` it is a TKAN-only feature -- requesting it for ``"mlp"``
    raises ``ValueError``, as does an unknown mode string.
    """
    window = cfg.effective_window()
    n_channels = len(window.channels)
    model_name = cfg.model.lower()

    if cfg.side_input and model_name not in _TKAN_MODELS:
        raise ValueError(
            f"side_input is implemented for tkan/sigkan only (the MLP baseline "
            f"is the side-free control); got model={cfg.model!r}"
        )

    if cfg.severity_mode not in _VALID_SEVERITY_MODES:
        raise ValueError(
            f"unknown severity_mode {cfg.severity_mode!r}; expected one of "
            f"{set(_VALID_SEVERITY_MODES)}"
        )
    if cfg.severity_mode == "quantile" and model_name not in _TKAN_MODELS:
        raise ValueError(
            f"severity_mode='quantile' is implemented for tkan/sigkan only (the "
            f"MLP baseline has a point severity head); got model={cfg.model!r}"
        )

    if model_name in _TKAN_MODELS:
        return TKANModel(
            input_size=n_channels,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            grid_size=cfg.grid_size,
            spline_order=cfg.spline_order,
            with_trajectory=cfg.with_trajectory,
            traj_steps=cfg.traj_steps,
            pool=cfg.pool,
            side_input=cfg.side_input,
            side_hidden=cfg.side_hidden,
            severity_mode=cfg.severity_mode,
            severity_quantiles=cfg.severity_quantiles,
        )
    if model_name == "mlp":
        return MLPBaseline(
            window_length=window.length,
            n_channels=n_channels,
            hidden_size=cfg.hidden_size,
        )
    raise ValueError(
        f"unknown model {cfg.model!r}; expected one of {{'tkan','sigkan','mlp'}}"
    )


# --------------------------------------------------------------------------- #
# Side-feature statistics (Gao-style process-parameter branch)
# --------------------------------------------------------------------------- #
def side_feature_stats(dataset, features) -> tuple[np.ndarray, np.ndarray]:
    """Window-weighted mean/std of per-file scalar ``features`` over ``dataset``.

    Iterates ``dataset._files`` (the per-file records built by
    ``Phase3WindowDataset``); each file contributes ``float(rec[feature])``
    weighted by its window count ``rec["windows"].shape[0]`` -- i.e. the
    per-window distribution the model actually sees. Returns ``(mean, std)``
    float64 arrays of length ``len(features)``; the std is the *population*
    std, floored at ``1e-8`` so a constant feature cannot divide by zero.

    Must be fed the TRAIN dataset only (val/test stats would leak). Raises
    ``ValueError`` if the dataset has no files.
    """
    files = list(getattr(dataset, "_files", []) or [])
    if not files:
        raise ValueError("side_feature_stats: dataset has no files")
    values = np.array(
        [[float(rec[f]) for f in features] for rec in files], dtype=np.float64
    )  # [n_files, S]
    weights = np.array(
        [rec["windows"].shape[0] for rec in files], dtype=np.float64
    )  # [n_files]
    mean = np.average(values, axis=0, weights=weights)
    var = np.average((values - mean) ** 2, axis=0, weights=weights)
    std = np.maximum(np.sqrt(var), 1e-8)
    return mean, std


# --------------------------------------------------------------------------- #
# Optimizer + lr schedule (fold-0 stability bundle)
# --------------------------------------------------------------------------- #
def build_optimizer(params, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Build the optimizer named by ``cfg.optimizer`` over ``params``.

    ``"adam"`` -> :class:`torch.optim.Adam`, ``"adamw"`` ->
    :class:`torch.optim.AdamW` (decoupled weight decay -- the principled way to
    regularize the spline weights). Both take ``lr=cfg.lr`` and
    ``weight_decay=cfg.weight_decay``; the defaults (``"adam"``, decay 0.0)
    reproduce the original hardcoded ``Adam(model.parameters(), lr=cfg.lr)``.
    """
    name = cfg.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(
        f"unknown optimizer {cfg.optimizer!r}; expected one of {{'adam','adamw'}}"
    )


def schedule_factor(
    epoch: int, total_epochs: int, schedule: str, warmup_epochs: int
) -> float:
    """Multiplicative lr factor for ``epoch`` (fed to ``LambdaLR``).

    Semantics (pinned by tests):

    * ``epoch < warmup_epochs`` (when warmup is on): linear ramp
      ``(epoch + 1) / warmup_epochs`` -- e.g. warmup 2 -> factors 0.5, 1.0.
    * ``"constant"``: 1.0 after warmup (the original fixed-lr behavior).
    * ``"cosine"``: half-cosine decay from 1.0 at ``epoch == warmup_epochs``
      toward ~0 at ``epoch == total_epochs`` over
      ``T = max(1, total_epochs - warmup_epochs)`` epochs.
    """
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        T = max(1, total_epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / T
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(
        f"unknown lr_schedule {schedule!r}; expected one of {{'constant','cosine'}}"
    )


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Phase3Trainer:
    """Adam training loop + per-file evaluation + trajectory stage-gate."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    # -- training ----------------------------------------------------------- #
    def fit(
        self,
        model,
        train_loader,
        val_loader,
        mlflow_run=None,
        output_dir: str | Path | None = None,
    ) -> dict:
        """Train ``model`` and return a history dict.

        Each epoch minimizes the mean :func:`total_loss` over ``train_loader``
        with Adam, then calls :meth:`evaluate` on ``val_loader``. A checkpoint is
        written every epoch to ``<output_dir>/checkpoints/epoch_{e}.pt``; the
        lowest-val-MAE state is mirrored to ``best.pt``, the highest-val-AUC
        state to ``best_auc.pt`` (the headline classification metric -- kept
        separately because AUC keeps climbing after MAE plateaus), and the last
        epoch to ``final.pt``. When ``mlflow_run`` is active, per-epoch train
        loss + val metrics are logged as MLflow metrics (step == epoch),
        including the per-epoch ``nonfinite_batches`` skip count.

        Non-finite guards (GL job 51641089 post-mortem): a batch whose loss is
        non-finite, or whose post-clip gradient norm is non-finite, is SKIPPED
        (grads zeroed, no ``optimizer.step()``) and counted, so one gradient
        blow-up cannot NaN-poison the whole model. ``train_loss`` averages only
        the finite, actually-stepped batches. If an epoch steps ZERO batches
        while skipping >0, a ``RuntimeError`` is raised (the model never
        trained that epoch; checkpointing would persist garbage).

        Returns ``{"train_loss": [...], "val": [metric_dict, ...],
        "nonfinite_batches": [int, ...]}`` (one entry per epoch each; the
        ``nonfinite_batches`` key is always present so summaries stay uniform).
        """
        _seed_everything(self.cfg.seed)
        epochs = self.cfg.effective_epochs()
        device = resolve_device(self.cfg.device)
        model.to(device)
        model = maybe_compile_cells(model, self.cfg, device)
        optimizer = build_optimizer(model.parameters(), self.cfg)
        # Warmup + schedule via LambdaLR: epoch e trains at factor(e) because
        # scheduler.step() runs once at the END of each epoch.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda e: schedule_factor(
                e, epochs, self.cfg.lr_schedule, self.cfg.warmup_epochs
            ),
        )

        ckpt_dir = None
        if output_dir is not None:
            ckpt_dir = Path(output_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        history: dict[str, list] = {"train_loss": [], "val": [], "nonfinite_batches": []}
        best_mae = float("inf")
        best_auc = float("-inf")

        for epoch in range(epochs):
            model.train()
            epoch_losses: list[float] = []
            # Skip-and-count guards (GL job 51641089 post-mortem): a real A40
            # fold-0 run died at the first evaluate() with sklearn "Input
            # contains NaN". The recurrent KAN-GRU's documented gradient
            # blow-ups can reach inf and yield NaN grads (inf*0 at saturated
            # tanh); clip_grad_norm_ (error_if_nonfinite=False) then computes
            # clip_coef = max_norm / (NaN + eps) and silently scales EVERY
            # parameter's grad by NaN, so one bad batch -> optimizer.step() ->
            # all weights NaN -> hours of silent garbage training. The guards
            # below skip such batches entirely so the weights stay finite.
            nonfinite_batches = 0
            n_batches = 0
            for batch in train_loader:
                n_batches += 1
                batch = _batch_to_device(batch, device)
                optimizer.zero_grad()
                out = model(batch)
                loss, _ = total_loss(out, batch, self.cfg.loss)
                # Cheap pre-backward guard: a NaN/inf loss can only produce
                # NaN/useless grads, so skip before backward (catches the blow-up
                # whether or not clipping is enabled).
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    nonfinite_batches += 1
                    continue
                loss.backward()
                # Recurrent KAN-GRU loss spikes are gradient blowups; clip the
                # global grad norm when enabled (grad_clip <= 0 keeps the
                # original unclipped behavior).
                if self.cfg.grad_clip > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.cfg.grad_clip
                    )
                    # We keep clip's default error_if_nonfinite=False and rely
                    # on this post-check instead: by the time the norm is
                    # observable, clip has ALREADY scaled the grads (by NaN when
                    # total_norm is NaN, to zero when it is inf), so zeroing the
                    # poisoned grads and skipping the step is the correct
                    # recovery. The inf case must also skip: zeroed grads still
                    # let AdamW apply weight decay / momentum drift on step.
                    if not torch.isfinite(total_norm):
                        optimizer.zero_grad()
                        nonfinite_batches += 1
                        continue
                optimizer.step()
                # Only finite, actually-stepped losses reach the epoch mean.
                epoch_losses.append(float(loss.detach()))

            if nonfinite_batches > 0:
                print(
                    f"[phase3_trainer] WARNING: epoch {epoch}: skipped "
                    f"{nonfinite_batches}/{n_batches} batches with non-finite "
                    f"loss or grad norm (weights kept finite; cf. GL job 51641089)",
                    file=sys.stderr,
                )
            if not epoch_losses and nonfinite_batches > 0:
                # Fail-loud floor: zero stepped batches means the model never
                # trained this epoch -- abort instead of checkpointing garbage
                # and burning the remaining wall time.
                raise RuntimeError(
                    f"epoch {epoch}: all {nonfinite_batches}/{n_batches} train "
                    f"batches were skipped as non-finite -- the model never "
                    f"trained this epoch (cf. GL job 51641089)."
                )
            history["nonfinite_batches"].append(nonfinite_batches)

            # The lr this epoch actually trained at -- read BEFORE scheduler.step().
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            val_metrics = self.evaluate(model, val_loader)
            history["train_loss"].append(mean_train_loss)
            history["val"].append(val_metrics)

            if mlflow_run is not None:
                self._log_epoch_metrics(
                    mean_train_loss,
                    val_metrics,
                    epoch,
                    lr=epoch_lr,
                    nonfinite_batches=nonfinite_batches,
                )

            if ckpt_dir is not None:
                torch.save(strip_compile_prefix(model.state_dict()), ckpt_dir / f"epoch_{epoch}.pt")
                val_mae = val_metrics.get("mae", float("nan"))
                # Treat NaN MAE as non-improving so a finite epoch always wins.
                if np.isfinite(val_mae) and val_mae < best_mae:
                    best_mae = val_mae
                    torch.save(strip_compile_prefix(model.state_dict()), ckpt_dir / "best.pt")
                # best_auc.pt: highest-val-AUC epoch. The headline metric vs the
                # MLP is AUC, and it keeps climbing after val MAE plateaus, so the
                # MAE-best (best.pt) is not the best classifier -- keep both.
                val_auc = val_metrics.get("auc", float("nan"))
                if np.isfinite(val_auc) and val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(strip_compile_prefix(model.state_dict()), ckpt_dir / "best_auc.pt")

            # Advance the lr schedule at the END of the epoch so the next epoch
            # trains at factor(epoch + 1). A no-op under the default constant
            # schedule with no warmup.
            scheduler.step()

        # Guarantee best.pt / best_auc.pt exist even if every epoch had NaN.
        if ckpt_dir is not None:
            if not (ckpt_dir / "best.pt").exists():
                torch.save(strip_compile_prefix(model.state_dict()), ckpt_dir / "best.pt")
            if not (ckpt_dir / "best_auc.pt").exists():
                torch.save(strip_compile_prefix(model.state_dict()), ckpt_dir / "best_auc.pt")
            torch.save(strip_compile_prefix(model.state_dict()), ckpt_dir / "final.pt")

        return history

    @staticmethod
    def _log_epoch_metrics(
        train_loss: float,
        val_metrics: dict,
        epoch: int,
        lr: float | None = None,
        nonfinite_batches: int | None = None,
    ) -> None:
        """Log per-epoch metrics to the active MLflow run (NaN-safe).

        ``lr`` is the learning rate the epoch actually trained at (read from the
        optimizer BEFORE the scheduler stepped), logged as metric ``"lr"``.
        ``nonfinite_batches`` is the epoch's skipped-batch count (the GL job
        51641089 guard), logged as metric ``"nonfinite_batches"`` so a poisoned
        run is visible in the tracking UI -- 0 on healthy epochs.
        """
        import mlflow

        if np.isfinite(train_loss):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
        if lr is not None and np.isfinite(lr):
            mlflow.log_metric("lr", float(lr), step=epoch)
        if nonfinite_batches is not None:
            mlflow.log_metric("nonfinite_batches", float(nonfinite_batches), step=epoch)
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                mlflow.log_metric(f"val_{key}", float(value), step=epoch)

    # -- evaluation --------------------------------------------------------- #
    def evaluate(self, model, val_loader) -> dict:
        """Per-file severity metrics over the whole validation loader.

        Collects per-window ``b_over_blim`` predictions + ``chatter_logit`` and
        the batch's ground-truth ``b_over_blim`` / ``label`` / ``file`` ids, then
        delegates to :func:`per_file_severity_metrics`. Runs under ``eval()`` +
        ``no_grad`` and restores ``train()`` before returning.
        """
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        b_pred: list[float] = []
        logit: list[float] = []
        b_true: list[float] = []
        label: list[float] = []
        file_ids: list = []

        with torch.no_grad():
            for batch in val_loader:
                batch = _batch_to_device(batch, device)
                out = model(batch)
                b_pred.extend(out["b_over_blim"].reshape(-1).tolist())
                logit.extend(out["chatter_logit"].reshape(-1).tolist())
                b_true.extend(batch["b_over_blim"].reshape(-1).tolist())
                label.extend(batch["label"].reshape(-1).tolist())
                file_ids.extend(list(batch["file"]))

        if was_training:
            model.train()

        return per_file_severity_metrics(
            np.asarray(b_pred),
            np.asarray(b_true),
            np.asarray(label),
            np.asarray(logit),
            file_ids,
        )

    # -- trajectory gate ---------------------------------------------------- #
    def run_gate(self, model, val_loader) -> dict:
        """Run the trajectory stage-gate, or report it disabled.

        If the model has no trajectory head
        (``getattr(model, "with_trajectory", False)`` is falsey) returns
        ``{"trajectory_enabled": False}``. Otherwise: from one validation batch
        pick a stable (``label==0``) and an unstable (``label==1``) sample, roll
        the model's trajectory for each, treat the rolled output as ``*_pred`` and
        the sample's own last ``traj_steps`` of the downstream channels
        (``traj_channels``) as ``*_true``, derive a simple onset index for each,
        and delegate to :func:`trajectory_gate_report`.
        """
        if not getattr(model, "with_trajectory", False):
            return {"trajectory_enabled": False}

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                report = self._gate_from_loader(model, val_loader)
        finally:
            if was_training:
                model.train()
        return report

    def _gate_from_loader(self, model, val_loader) -> dict:
        """Build the stable/unstable trajectory pair + onsets and grade them.

        Scans the validation loader for the first stable (``label==0``) and the
        first unstable (``label==1``) single-window sample (they may live in
        different batches), rolls the model's trajectory for each, and grades the
        rolled vs. observed-tail pair.
        """
        traj_steps = int(getattr(model, "traj_steps", self.cfg.traj_steps))
        traj_channels = list(getattr(model, "traj_channels", (0, 2)))

        stable = self._sample_for_label(val_loader, label=0.0)
        unstable = self._sample_for_label(val_loader, label=1.0)
        if stable is None or unstable is None:
            # The val loader lacked one of the two classes; cannot grade.
            return {
                "trajectory_enabled": True,
                "passed": False,
                "reason": "missing_stable_or_unstable_sample",
            }

        stable_pred, stable_true = self._roll_one(model, stable, traj_steps, traj_channels)
        unstable_pred, unstable_true = self._roll_one(
            model, unstable, traj_steps, traj_channels
        )

        report = trajectory_gate_report(
            stable_true=stable_true,
            stable_pred=stable_pred,
            unstable_true=unstable_true,
            unstable_pred=unstable_pred,
            true_onset=_onset_index(unstable_true),
            predicted_onset=_onset_index(unstable_pred),
        )
        report["trajectory_enabled"] = True
        return report

    @staticmethod
    def _sample_for_label(val_loader, label: float):
        """Return a single-window batch dict for the first row matching ``label``.

        The window keeps a batch axis of 1 so it can be fed straight to the model.
        Scalar side features (``omega``/``tpf``) are carried along when the
        collate provides them so side-input models can run the gate too.
        Returns ``None`` if no such row exists in the loader.
        """
        for batch in val_loader:
            labels = batch["label"].reshape(-1)
            hits = torch.nonzero(labels == label, as_tuple=False)
            if hits.numel() == 0:
                continue
            i = int(hits[0].item())
            single = {
                "window": batch["window"][i : i + 1],
                "label": batch["label"][i : i + 1],
                "tau": batch["tau"][i : i + 1],
                "b_over_blim": batch["b_over_blim"][i : i + 1],
                "sample_weight": batch["sample_weight"][i : i + 1],
            }
            for key in ("omega", "tpf"):
                if key in batch:
                    single[key] = batch[key][i : i + 1]
            return single
        return None

    @staticmethod
    def _roll_one(model, single: dict, traj_steps: int, traj_channels: list):
        """Roll the trajectory head for one sample; return (pred, true) arrays.

        ``pred`` is the model's rolled trajectory ``[traj_steps, len(traj_channels)]``;
        ``true`` is the sample's own last ``traj_steps`` of the same downstream
        channels.
        """
        device = next(model.parameters()).device
        single = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in single.items()
        }
        out = model(single)
        traj_pred = out["trajectory"][0].cpu().numpy()  # [traj_steps, len(tc)]
        window = single["window"]  # [1, L, C]
        true_tail = window[0, -traj_steps:, traj_channels].cpu().numpy()
        return traj_pred, true_tail


def _onset_index(series: np.ndarray) -> float:
    """First step index whose abs value exceeds ``mean + 2*std`` (fallback 0).

    ``series`` is ``[traj_steps, n_channels_out]``; we collapse channels by the
    max abs activation per step, then threshold against that scalar series'
    ``mean + 2*std``.
    """
    arr = np.abs(np.asarray(series, dtype=float))
    if arr.ndim > 1:
        arr = arr.max(axis=tuple(range(1, arr.ndim)))
    if arr.size == 0:
        return 0.0
    threshold = float(arr.mean() + 2.0 * arr.std())
    above = np.nonzero(arr > threshold)[0]
    return float(above[0]) if above.size else 0.0


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    """Recursively coerce dataclasses / numpy / tensors into JSON-safe values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if torch.is_tensor(obj):
        return _json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def _resolve_tracking_uri(tracking_uri: str | None) -> str:
    """Use the explicit URI or fall back to the project config's phase3 URI."""
    if tracking_uri:
        return tracking_uri
    return load_config()["phase3"]["mlflow"]["tracking_uri"]


def _excluded_reason_counts(table) -> dict[str, int]:
    """Count supervised-excluded rows of ``table`` grouped by ``exclude_reason``.

    Returns ``{reason: count}`` over rows where ``exclude_supervised`` is truthy.
    If the table lacks an ``exclude_reason`` column (e.g. older fixtures) the
    excluded rows are grouped under ``"unspecified"`` so the count still totals
    ``n_excluded``. Missing ``exclude_supervised`` yields an empty dict.
    """
    import pandas as pd

    if "exclude_supervised" not in getattr(table, "columns", []):
        return {}
    excluded = table[table["exclude_supervised"].astype(bool)]
    if "exclude_reason" in excluded.columns:
        reasons = excluded["exclude_reason"]
    else:
        reasons = pd.Series(["unspecified"] * len(excluded), index=excluded.index)
    return {str(k): int(v) for k, v in reasons.value_counts().items()}


def _build_preprocessing(
    cfg: "TrainConfig",
    norm: NormStats,
    table,
    train_loader,
    val_loader,
    side_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Assemble the per-fold ``preprocessing.json`` payload.

    Captures everything a trained checkpoint needs to recover its z-score on
    new/raw data: the TRAIN-only per-channel ``norm`` (mean/std), the channel
    order, the window geometry, the temporal pool factor and fold index, plus
    train/val window+file counts and the supervised-exclusion breakdown. The
    leakage guard already ran inside ``build_fold_loaders``, so ``leakage_verified``
    is always ``True`` here.

    ``side_stats`` (the TRAIN-only ``(mean, std)`` from
    :func:`side_feature_stats`, present iff ``cfg.side_input``) adds a
    ``"side"`` block with the feature names + stats. The key is ABSENT when the
    side branch is off -- downstream XAI detects side checkpoints by its
    presence.

    A non-default ``cfg.severity_mode`` adds a ``"severity"`` block
    (``{"mode", "quantiles"}``) for the same reason: the mode changes the
    severity head's output width, so a checkpoint cannot be rebuilt without it.
    The key is likewise ABSENT for the default point head.
    """
    window = cfg.effective_window()
    payload = {
        "norm": norm.to_dict(),
        "channels": list(window.channels),
        "window": {"length": window.length, "stride": window.stride},
        "pool": cfg.pool,
        "fold": cfg.fold,
        "counts": {
            "n_train_windows": len(train_loader.dataset),
            "n_val_windows": len(val_loader.dataset),
            "n_train_files": len(train_loader.dataset._files),
            "n_val_files": len(val_loader.dataset._files),
            "n_excluded": int(table["exclude_supervised"].astype(bool).sum())
            if "exclude_supervised" in getattr(table, "columns", [])
            else 0,
            "excluded_reasons": _excluded_reason_counts(table),
        },
        "leakage_verified": True,
        "seed": cfg.seed,
    }
    if side_stats is not None:
        mean, std = side_stats
        payload["side"] = {
            "features": list(DEFAULT_SIDE_FEATURES),
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
        }
    if cfg.severity_mode != "huber":
        # A non-default severity head changes the head's OUTPUT WIDTH, so a
        # checkpoint cannot be rebuilt without knowing the mode and the exact
        # quantile levels (the levels are unrecoverable from the state_dict --
        # only their count is). Absent == the default point head, mirroring how
        # the "side" block marks side-input checkpoints.
        payload["severity"] = {
            "mode": cfg.severity_mode,
            "quantiles": [float(q) for q in cfg.severity_quantiles],
        }
    return payload


def _best_epoch(val_history: list[dict], fallback: dict) -> tuple[int, dict]:
    """Return ``(epoch_index, metrics)`` of the lowest finite val-MAE epoch.

    This mirrors the ``best.pt`` selection rule in :meth:`Phase3Trainer.fit`
    (lowest finite val MAE wins). If no epoch has a finite MAE, returns
    ``(-1, fallback)`` so the caller still has a sane metrics dict.
    """
    best_idx = -1
    best_mae = float("inf")
    for i, metrics in enumerate(val_history):
        mae = metrics.get("mae", float("nan"))
        if isinstance(mae, (int, float)) and np.isfinite(mae) and mae < best_mae:
            best_mae = mae
            best_idx = i
    if best_idx < 0:
        return -1, fallback
    return best_idx, val_history[best_idx]


def _best_auc_epoch(val_history: list[dict], fallback: dict) -> tuple[int, dict]:
    """Return ``(epoch_index, metrics)`` of the highest finite val-AUC epoch.

    Mirrors the ``best_auc.pt`` selection rule in :meth:`Phase3Trainer.fit`
    (highest finite val AUC wins -- the headline classification metric). If no
    epoch has a finite AUC, returns ``(-1, fallback)``.
    """
    best_idx = -1
    best_auc = float("-inf")
    for i, metrics in enumerate(val_history):
        auc = metrics.get("auc", float("nan"))
        if isinstance(auc, (int, float)) and np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_idx = i
    if best_idx < 0:
        return -1, fallback
    return best_idx, val_history[best_idx]


def train_one_fold(
    cfg: TrainConfig,
    table,
    folds_json,
    npz_root,
    output_dir,
    tracking_uri: str | None = None,
) -> dict:
    """Train one fold end-to-end and return a summary dict.

    Steps: seed -> build leakage-safe fold loaders (``build_fold_loaders`` asserts
    no mirror leakage internally) -> build the model -> open an MLflow run on a
    ``file:`` URI -> log key params -> fit -> log final val metrics -> run the
    trajectory gate -> write ``<run_dir>/preprocessing.json`` (the TRAIN-only
    z-score + window/channel geometry a checkpoint needs to score raw data) and
    ``<run_dir>/summary.json``, logging both as MLflow artifacts -> prune periodic
    checkpoints via :func:`apply_checkpoint_retention`. The returned dict carries
    ``val``, ``gate``, ``history`` / ``history_tail``, ``preprocessing``,
    ``run_id``, ``run_dir``, ``checkpoints_dir`` and the ``retention`` plan.
    """
    import mlflow

    _seed_everything(cfg.seed)

    device = resolve_device(cfg.device)
    train_loader, val_loader, norm = build_fold_loaders(
        table,
        folds_json,
        cfg.fold,
        npz_root,
        cfg.effective_window(),
        cfg.batch_size,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    # TRAIN-only side-feature stats (window-weighted) for the Gao side branch;
    # computed once, baked into both the model buffers and preprocessing.json.
    side_stats = (
        side_feature_stats(train_loader.dataset, DEFAULT_SIDE_FEATURES)
        if cfg.side_input
        else None
    )
    # Capture the TRAIN-only z-score so a checkpoint can be reapplied to raw data
    # (build_fold_loaders asserted no mirror leakage, hence leakage_verified).
    preprocessing = _build_preprocessing(
        cfg, norm, table, train_loader, val_loader, side_stats=side_stats
    )
    model = build_model(cfg)
    if side_stats is not None:
        # Bake the TRAIN-fold stats into the state_dict buffers BEFORE fit so
        # every checkpoint is self-contained (val stats would leak; never used).
        model.set_side_norm(*side_stats)
    trainer = Phase3Trainer(cfg)

    mlflow.set_tracking_uri(_resolve_tracking_uri(tracking_uri))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        run_dir = output_dir / run_id
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        mlflow.log_params(
            {
                "model": cfg.model,
                "epochs": cfg.effective_epochs(),
                "lr": cfg.lr,
                "lr_schedule": cfg.lr_schedule,
                "warmup_epochs": cfg.warmup_epochs,
                "grad_clip": cfg.grad_clip,
                "optimizer": cfg.optimizer,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "fold": cfg.fold,
                "seed": cfg.seed,
                "window_length": cfg.effective_window().length,
                "window_stride": cfg.effective_window().stride,
                "hidden_size": cfg.hidden_size,
                "with_trajectory": cfg.with_trajectory,
                "side_input": cfg.side_input,
                "side_hidden": cfg.side_hidden,
                "severity_mode": cfg.severity_mode,
                "severity_quantiles": list(cfg.severity_quantiles),
                "device": str(device),
                "num_workers": cfg.num_workers,
            }
        )

        history = trainer.fit(
            model, train_loader, val_loader, mlflow_run=run, output_dir=run_dir
        )
        val_metrics = history["val"][-1] if history["val"] else trainer.evaluate(model, val_loader)

        # Best epoch == the one mirrored to best.pt (lowest finite val MAE). The
        # final-epoch ``val`` block alone is misleading because per-epoch metrics
        # are noisy; this surfaces the metrics of the checkpoint actually kept.
        best_epoch, best_val = _best_epoch(history["val"], val_metrics)
        # best_auc.pt's epoch: the headline-AUC peak (kept alongside best.pt so
        # the number reported vs the MLP is the actual peak, not the MAE-best).
        best_auc_epoch, best_auc_val = _best_auc_epoch(history["val"], val_metrics)

        for key, value in val_metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                mlflow.log_metric(f"final_val_{key}", float(value))
        for key, value in best_val.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                mlflow.log_metric(f"best_val_{key}", float(value))
        for key, value in best_auc_val.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                mlflow.log_metric(f"best_auc_val_{key}", float(value))

        gate = trainer.run_gate(model, val_loader)
        mlflow.log_param("trajectory_enabled", bool(gate.get("trajectory_enabled", False)))
        if "passed" in gate:
            mlflow.log_metric("gate_passed", float(bool(gate["passed"])))
        if "min_r2" in gate and np.isfinite(gate["min_r2"]):
            mlflow.log_metric("gate_min_r2", float(gate["min_r2"]))

        retention = apply_checkpoint_retention(ckpt_dir, dry_run=False)

        # Persist + log the per-fold preprocessing so a checkpoint's z-score is
        # recoverable on new/raw data (state_dict alone cannot reconstruct it).
        preprocessing_path = run_dir / "preprocessing.json"
        preprocessing_path.write_text(json.dumps(_json_safe(preprocessing), indent=2))
        mlflow.log_artifact(str(preprocessing_path))

        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "checkpoints_dir": str(ckpt_dir),
            "model": cfg.model,
            "fold": cfg.fold,
            "config": _json_safe(cfg),
            "preprocessing": _json_safe(preprocessing),
            "val": _json_safe(val_metrics),
            "best_epoch": best_epoch,
            "best_val": _json_safe(best_val),
            "best_auc_epoch": best_auc_epoch,
            "best_auc_val": _json_safe(best_auc_val),
            "gate": _json_safe(gate),
            "history": _json_safe(history),
            "history_tail": _json_safe(
                {"train_loss": history["train_loss"][-1:], "val": history["val"][-1:]}
            ),
            "retention": {
                "keep": list(retention.keep),
                "delete": list(retention.delete),
            },
        }

        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(summary_path))

    return summary
