"""U14-B: do recurring activation *modes* exist, or does random init do this too?

The question this module answers is narrow and deliberately hostile to itself:
**does a trained TKAN learn a small set of recurring edge-activation shapes,
beyond what randomly initialised weights of the same architecture produce by
chance?** Two Phase 3 statistics have already collapsed under exactly this kind
of control -- a TSR delay-lock that scored *higher* on an untrained net, and a
band-notch contrast that reported a solver failure as a measurement -- so every
statistic here is built so the untrained arm runs through the identical code
path, and so a missing measurement can never be read as a pass.

What this module will and will not claim
----------------------------------------
``grid_size=5`` + ``spline_order=3`` gives ``n_coeff = 8`` basis functions, and
inside one :class:`~pri2m.models.kan.KANLinear` every edge curve is ``B @ c``
over an **identical** basis ``B``. A PCA over sampled curves is therefore a PCA
over 8-dimensional coefficient vectors: it *cannot* exceed 8 components, and a
low-dimensional spectrum is partly architecture, not learning. The spectrum is
reported (:func:`spectrum_diagnostic`) under a name that says so, with
``is_finding=False``. **"k modes explain X% of variance" is never a finding
here.** The only finding this module can produce is the trained-versus-untrained
margin from :func:`mode_verdict`.

Design constraints, and where each is enforced
----------------------------------------------
untrained null is mandatory
    :func:`reinitialised_copy` / :func:`assert_reinitialised` (the two guards
    from ``scripts/tsr_control_phase3.py``), driven by
    :func:`untrained_replicate`. :func:`mode_verdict` returns UNDECIDED unless
    at least ``min_seeds`` (default 20) untrained seeds are supplied -- the
    null's spread is the denominator of the whole test, and 5 seeds do not
    measure a spread.
matched N
    :func:`match_gate_sizes`. If an absolute importance floor passes 90% of
    trained edges and 5% of untrained ones the two arms are not the same
    population, so every comparison is made after truncating both arms to the
    common count (by importance). Raw per-arm pass fractions are reported too.
gate before normalisation
    :func:`gate_by_importance` refuses a normalised bundle and
    :func:`normalise_curves` refuses an ungated one. A shape-normalised dead
    edge is indistinguishable from a shape-normalised dominant one.
spline-only clustering, base fraction reported
    Curves are extracted with ``component="spline"``; the base path is exactly
    ``base_weight[o,i] * SiLU(x)``, a one-parameter family with no shape
    information. Each record carries ``base_fraction`` so any claim about what
    the network *computes* can be referred back to the full curve.
stratification
    :func:`edge_stratum` / :func:`edge_identity` / :func:`require_fold_stable`.
    Only ``encoder.layers.0.kan_{z,r,h}`` x_t / x_delay inputs and the
    ``severity.{cls,reg}_head`` side block carry fold-stable indices. Hidden-fed
    edges get ``identity=None`` and any attempt to compare them by index raises.

Everything here is a pure function over an already-loaded model plus a batch:
no file IO, plain dicts out (:func:`jsonable` makes the top-level result
JSON-safe, mapping every non-finite float to ``None`` so an unmeasurable value
can never be read as a number).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence

import numpy as np
import torch

from pri2m.xai.activations import collect_activations
from pri2m.xai.semantics import edge_index_map

# --------------------------------------------------------------------------- #
# Strata
# --------------------------------------------------------------------------- #

#: Strata whose *input* indices mean the same thing in every independently
#: trained fold. Everything else is comparable only as a shape population.
FOLD_STABLE_STRATA: frozenset[str] = frozenset(
    {"encoder.L0.x_t", "encoder.L0.x_delay", "severity.side"}
)

_ENCODER_RE = re.compile(r"^encoder\.layers\.(\d+)\.(kan_[zrh])$")
_HIDDEN_BLOCKS = frozenset({"h_prev", "r*h_prev", "hidden"})


def is_fold_stable(stratum: str) -> bool:
    """Is ``stratum``'s input index the same physical quantity in every fold?"""
    return stratum in FOLD_STABLE_STRATA


def edge_stratum(kan_name: str, entry: dict) -> str:
    """Stratum key for one input edge of one ``KANLinear``.

    Parameters
    ----------
    kan_name:
        Qualified module name, e.g. ``"encoder.layers.0.kan_z"``.
    entry:
        One ``{"index", "block", "name"}`` record from
        :func:`pri2m.xai.semantics.edge_index_map`.

    Returns
    -------
    str
        e.g. ``"encoder.L0.x_delay"``, ``"encoder.L1.x_t"``, ``"severity.side"``.

    Notes
    -----
    The layer index is part of the key on purpose. ``block == "x_t"`` is a raw
    physical channel **only on layer 0**; on layer ``k > 0`` the same block name
    holds the previous layer's hidden state (``L{k-1}_h*``), whose unit indices
    are permutation-arbitrary across folds. Keying on the block alone is the
    obvious implementation and it is wrong.
    """
    m = _ENCODER_RE.match(kan_name)
    if m is not None:
        layer = int(m.group(1))
        block = entry["block"]
        if block in _HIDDEN_BLOCKS:
            return f"encoder.L{layer}.h_prev"
        return f"encoder.L{layer}.{block}"
    if kan_name.startswith("severity."):
        return f"severity.{entry['block']}"
    return f"other.{entry['block']}"


def edge_identity(kan_name: str, entry: dict, out: int, stratum: str) -> str | None:
    """Cross-fold identity of an edge, or ``None`` if it has none.

    Returns
    -------
    str or None
        ``None`` for every hidden-fed edge -- those are comparable as a shape
        population and never by index.

    Notes
    -----
    Encoder identities deliberately omit ``out``: an encoder gate's out-units
    are hidden units, so ``out`` is permutation-arbitrary across independently
    trained folds even when ``in`` is not ("comparable on ``in`` but never on
    ``out``"). The severity side block is the one place ``out`` is stable too
    (``cls_head`` has a single logit; ``reg_head``'s outputs are fixed
    quantile/point slots), so it is kept there.
    """
    if not is_fold_stable(stratum):
        return None
    if stratum == "severity.side":
        return f"{kan_name}|{entry['name']}|out={int(out)}"
    return f"{kan_name}|{entry['name']}"


def require_fold_stable(bundle: dict) -> None:
    """Raise unless ``bundle``'s stratum may be compared by index across folds.

    Guards the one mistake that would silently produce a meaningless number:
    matching hidden-unit ``h7`` in fold 0 against hidden-unit ``h7`` in fold 3.
    Independently trained folds have no reason to order their hidden units the
    same way.
    """
    if not bundle.get("fold_stable", False):
        raise ValueError(
            f"stratum {bundle.get('stratum')!r} is not fold-stable: its indices are "
            "permutation-arbitrary across independently trained folds, so its edges "
            "may only be compared as a shape population, never by identity "
            f"(fold-stable strata: {sorted(FOLD_STABLE_STRATA)})"
        )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _component_stds(kan, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge std over the data of the base and spline contributions.

    Returns ``(base_std[out, in], spline_std[out, in])``.

    Computed in closed form rather than by materialising the ``[N, out, in]``
    contribution tensor that :func:`pri2m.xai.activations._edge_importance`
    builds: on the real model that tensor is ~19M floats per gate, and this
    function is called once per fold *per untrained seed*. Both identities are
    exact, not approximations:

    * ``base[n,o,i] = base_weight[o,i] * SiLU(x[n,i])``, so
      ``std_n = |base_weight[o,i]| * std_n(SiLU(x[:,i]))``.
    * ``spline[n,o,i] = scale * sum_c B[n,i,c] W[o,i,c]``, so
      ``var_n = scale^2 * W[o,i,:] Cov_n(B[:,i,:]) W[o,i,:]^T``.
    """
    with torch.no_grad():
        base_x = kan.base_activation(x)                      # [N, in]
        base_std = (
            kan.base_weight.abs() * base_x.std(dim=0, unbiased=False).unsqueeze(0)
        )                                                    # [out, in]

        basis = kan.b_splines(x)                             # [N, in, n_coeff]
        centred = basis - basis.mean(dim=0, keepdim=True)
        cov = torch.einsum("nic,nid->icd", centred, centred) / basis.shape[0]
        w = kan.spline_weight                                # [out, in, n_coeff]
        var = torch.einsum("oic,icd,oid->oi", w, cov, w)
        spline_std = var.clamp_min(0.0).sqrt() * abs(kan.spline_scale)
        return (base_std.cpu().numpy(), spline_std.cpu().numpy())


def extract_stratified_curves(
    model,
    stats: dict,
    sem_map: dict | None = None,
    *,
    n: int = 64,
    component: str = "spline",
    strata: Sequence[str] | None = None,
) -> dict[str, dict]:
    """Sample every edge curve of ``model``, bucketed into non-poolable strata.

    Parameters
    ----------
    model:
        A loaded Phase 3 model (already forwarded through ``stats``).
    stats:
        Output of :func:`pri2m.xai.activations.collect_activations` -- supplies
        the data-aware ``importance[o, i]``, the visited range, and the live
        ``KANLinear`` module for each layer.
    sem_map:
        Output of :func:`pri2m.xai.semantics.edge_index_map`; computed from
        ``model`` when omitted.
    n:
        Sweep points per curve. Curves are sampled on the layer's shared
        ``grid_range`` grid, **not** on each edge's own visited range: inside a
        ``KANLinear`` every curve is ``B @ c`` over one identical basis, so a
        shared grid makes Euclidean/cosine curve distance a genuine distance in
        coefficient space. Per-edge grids would make the distances
        incommensurable. Each record still carries ``visited_lo``/``visited_hi``
        so a caller can report coverage.
    component:
        Passed to :meth:`pri2m.models.kan.KANLinear.edge_function`. Defaults to
        ``"spline"``: the base path is a one-parameter family and would form one
        large trivial cluster.
    strata:
        Which strata to keep. ``None`` keeps **all** of them (including the
        hidden-fed ones, which are legitimate as a shape population).

    Returns
    -------
    dict
        ``{stratum: bundle}``. A bundle holds ``curves[E, n]`` (float32),
        ``importance[E]``, ``base_fraction[E]``, ``records`` (per-edge metadata),
        plus ``gate``/``normalised`` provenance slots that later stages fill in.
        Bundles are per-stratum by construction, which is what makes pooling a
        hidden-fed edge with a physical-channel edge impossible rather than
        merely discouraged.
    """
    if sem_map is None:
        sem_map = edge_index_map(model)
    wanted = None if strata is None else set(strata)

    acc: dict[str, dict] = {}
    for kan_name, stat in stats.items():
        kan = stat.get("module")
        if kan is None:
            continue
        entries = sem_map.get(kan_name)
        if not entries:
            continue
        keep = [e for e in entries
                if wanted is None or edge_stratum(kan_name, e) in wanted]
        if not keep:
            continue

        curves = kan.edge_function(n, component=component)   # [out, in, n]
        xs = curves["x"]
        imp = np.asarray(stat["importance"], dtype=np.float64)
        x_data = torch.as_tensor(np.asarray(stat["inputs"]), dtype=torch.float32)
        base_std, spline_std = _component_stds(kan, x_data)
        denom = base_std + spline_std
        # 0/0 for a doubly-dead edge: report 0.0 rather than NaN, and let the
        # importance gate remove it (its importance is 0 too).
        base_frac = np.divide(base_std, denom, out=np.zeros_like(base_std),
                              where=denom > 0)
        lo = np.asarray(stat["visited_lo"], dtype=np.float64)
        hi = np.asarray(stat["visited_hi"], dtype=np.float64)
        sig = (float(kan.grid_range[0]), float(kan.grid_range[1]),
               int(kan.grid_size), int(kan.spline_order))

        for entry in keep:
            i = int(entry["index"])
            stratum = edge_stratum(kan_name, entry)
            bucket = acc.setdefault(stratum, {
                "stratum": stratum,
                "fold_stable": is_fold_stable(stratum),
                "x": xs,
                "curves": [],
                "importance": [],
                "base_fraction": [],
                "records": [],
                "n_coeff": int(kan.n_coeff),
                "grid_range": [sig[0], sig[1]],
                "component": component,
                "_sig": sig,
            })
            if bucket["_sig"] != sig:
                # Curves from different bases are not in the same vector space;
                # clustering them together would be meaningless arithmetic.
                raise ValueError(
                    f"stratum {stratum!r} would pool KANLinear layers with different "
                    f"spline bases ({bucket['_sig']} vs {sig}); refusing to cluster "
                    "curves that do not live in the same function space"
                )
            for o in range(kan.out_features):
                bucket["curves"].append(curves["curves"][o, i])
                bucket["importance"].append(float(imp[o, i]))
                bucket["base_fraction"].append(float(base_frac[o, i]))
                bucket["records"].append({
                    "kan": kan_name,
                    "out": int(o),
                    "in": i,
                    "block": entry["block"],
                    "feature": entry["name"],
                    "stratum": stratum,
                    "identity": edge_identity(kan_name, entry, o, stratum),
                    "importance": float(imp[o, i]),
                    "base_fraction": float(base_frac[o, i]),
                    "visited_lo": float(lo[i]),
                    "visited_hi": float(hi[i]),
                })

    out: dict[str, dict] = {}
    for stratum, bucket in acc.items():
        bucket.pop("_sig")
        bucket["curves"] = np.asarray(bucket["curves"], dtype=np.float32)
        bucket["importance"] = np.asarray(bucket["importance"], dtype=np.float64)
        bucket["base_fraction"] = np.asarray(bucket["base_fraction"], dtype=np.float64)
        bucket["n_edges"] = int(bucket["curves"].shape[0])
        bucket["gate"] = None
        bucket["normalised"] = None
        bucket["n_degenerate_dropped"] = 0
        out[stratum] = bucket
    return out


def _subset(bundle: dict, idx: np.ndarray) -> dict:
    """Copy ``bundle`` keeping only the edges at ``idx`` (order preserved)."""
    idx = np.asarray(idx, dtype=np.int64)
    new = dict(bundle)
    new["curves"] = bundle["curves"][idx]
    new["importance"] = bundle["importance"][idx]
    new["base_fraction"] = bundle["base_fraction"][idx]
    new["records"] = [bundle["records"][int(j)] for j in idx]
    new["n_edges"] = int(len(idx))
    return new


# --------------------------------------------------------------------------- #
# Gating (raw importance) -- must precede any shape normalisation
# --------------------------------------------------------------------------- #

def gate_by_importance(
    bundle: dict,
    *,
    quantile: float | None = None,
    min_importance: float | None = None,
    top_n: int | None = None,
) -> dict:
    """Drop near-dead edges on **raw data-aware importance**.

    Exactly one of ``quantile`` / ``min_importance`` / ``top_n`` must be given.

    Parameters
    ----------
    quantile:
        Keep edges at or above this quantile of the stratum's own importance
        (``0.9`` keeps the top 10%). Relative: it passes the same *fraction* of
        every arm by construction, so it cannot expose an arm-to-arm scale
        difference -- read the reported ``threshold`` for that.
    min_importance:
        Absolute floor. This is the scientifically honest gate and the one that
        makes :func:`match_gate_sizes` necessary: the trained and untrained arms
        sit at different importance scales, so an absolute floor passes very
        different counts.
    top_n:
        Keep the ``top_n`` most important edges. Used by
        :func:`match_gate_sizes`.

    Returns
    -------
    dict
        A new bundle carrying a ``"gate"`` record
        (``mode``/``threshold``/``n_before``/``n_after``/``pass_fraction``/
        ``importance_median``).

    Notes
    -----
    Gating on the *unnormalised* importance is the whole point. Once a curve is
    shape-normalised, a dead edge and a dominant edge are the same unit vector;
    a gate applied afterwards would be selecting on noise. Calling this on an
    already-normalised bundle raises.
    """
    if bundle.get("normalised"):
        raise ValueError(
            "refusing to gate an already shape-normalised bundle: after "
            "normalisation a dead edge is indistinguishable from a dominant one, "
            "so the importance gate must run first"
        )
    given = [x is not None for x in (quantile, min_importance, top_n)]
    if sum(given) != 1:
        raise ValueError(
            "pass exactly one of quantile=, min_importance= or top_n= "
            f"(got {sum(given)})"
        )

    imp = bundle["importance"]
    n_before = int(bundle["n_edges"])
    # Hard floor, applied in EVERY mode: importance is a std over the realised
    # data, so exactly 0 means the edge's contribution never moves -- it is dead
    # whatever quantile it happens to sit at. Without this a stratum that is
    # entirely dead in the data (e.g. side inputs that are constant across the
    # batch) sails through a quantile gate at threshold 0 and then produces a
    # perfectly ordinary-looking mode score from curves the data never probes.
    alive = imp > 0.0
    if quantile is not None:
        thresh = float(np.quantile(imp, float(quantile))) if n_before else 0.0
        keep = np.flatnonzero((imp >= thresh) & alive)
        mode = "quantile"
        param: float = float(quantile)
    elif min_importance is not None:
        thresh = float(min_importance)
        keep = np.flatnonzero((imp >= thresh) & alive)
        mode = "min_importance"
        param = float(min_importance)
    else:
        order = np.argsort(-imp, kind="stable")
        order = order[alive[order]][: int(top_n)]
        keep = np.sort(order)
        thresh = float(imp[keep].min()) if len(keep) else float("nan")
        mode = "top_n"
        param = float(top_n)

    new = _subset(bundle, keep)
    new["gate"] = {
        "mode": mode,
        "param": param,
        "threshold": thresh,
        "n_before": n_before,
        "n_after": int(len(keep)),
        "pass_fraction": float(len(keep) / n_before) if n_before else 0.0,
        "importance_median": float(np.median(imp)) if n_before else None,
        # Sentinel, reported not averaged: how much of this stratum the data
        # never moved at all.
        "n_zero_importance": int((~alive).sum()),
        "matched_n": None,
    }
    return new


def match_gate_sizes(bundles: Sequence[dict]) -> list[dict]:
    """Truncate every arm to the common edge count, keeping the most important.

    The plan's exact failure mode: if the gate passes 90% of trained edges and
    5% of untrained ones, the two arms are **not the same population** and any
    trained-vs-untrained comparison of a population statistic is meaningless.
    Every comparison in this module is made after this call.

    Returns
    -------
    list[dict]
        New bundles, all with ``n_edges == min(n_edges)`` and
        ``gate["matched_n"]`` set.
    """
    bundles = list(bundles)
    if not bundles:
        return []
    for b in bundles:
        if b.get("gate") is None:
            raise ValueError(
                "match_gate_sizes expects gated bundles; run gate_by_importance "
                "first so the matched population is drawn from a defined gate"
            )
    n = int(min(int(b["n_edges"]) for b in bundles))
    out = []
    for b in bundles:
        order = np.argsort(-b["importance"], kind="stable")[:n]
        new = _subset(b, np.sort(order))
        new["gate"] = {**b["gate"], "matched_n": n, "n_after": int(new["n_edges"])}
        out.append(new)
    return out


# --------------------------------------------------------------------------- #
# Shape normalisation
# --------------------------------------------------------------------------- #

def normalise_curves(bundle: dict, *, center: bool = True) -> dict:
    """Centre and unit-normalise each curve so only its *shape* remains.

    Parameters
    ----------
    center:
        Subtract each curve's mean over the sweep. A constant offset is absorbed
        downstream by other edges and the head bias, so it carries no shape
        information; removing it also removes the one direction (the B-spline
        partition-of-unity vector) that every curve shares.

    Returns
    -------
    dict
        A new bundle with unit-norm curves, ``normalised`` set, and
        ``n_degenerate_dropped`` counting curves whose norm was numerically
        zero. Those are **dropped and counted, never rescaled**: a zero curve
        divided by its own (zero) norm would either produce NaN or, with a
        naive epsilon, an ordinary-looking unit vector -- a dead edge
        masquerading as a shape. Idempotent.

    Raises
    ------
    ValueError
        If the bundle has not been through :func:`gate_by_importance`.
    """
    if bundle.get("normalised"):
        return bundle
    if bundle.get("gate") is None:
        raise ValueError(
            "refusing to shape-normalise an ungated bundle: the importance gate "
            "must run on raw data-aware importance first, because a "
            "shape-normalised dead edge looks exactly like a dominant one"
        )
    curves = np.asarray(bundle["curves"], dtype=np.float64)
    if center and curves.size:
        curves = curves - curves.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(curves, axis=1) if curves.size else np.zeros(0)
    scale = float(norms.max()) if norms.size else 0.0
    thresh = max(1e-12, 1e-9 * scale)
    keep = np.flatnonzero(norms > thresh)

    new = _subset(bundle, keep)
    new["curves"] = (curves[keep] / norms[keep][:, None]).astype(np.float64)
    new["normalised"] = "center+unit_l2" if center else "unit_l2"
    new["n_degenerate_dropped"] = int(bundle["n_edges"] - len(keep))
    return new


# --------------------------------------------------------------------------- #
# Prototypes (k-means on the unit sphere) and the spectrum diagnostic
# --------------------------------------------------------------------------- #

def _dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``A @ B`` with the inputs checked and numpy's spurious BLAS flags muted.

    numpy >= 2.1 surfaces the floating-point status flags left behind by the
    vectorised BLAS kernel, which sets divide/overflow/invalid on padding lanes
    even when every input and output is finite (verified against ``einsum`` on
    this data). Muting the flags blindly would hide a genuine NaN, so the
    operands are checked explicitly first -- a NaN reaching here is a real bug
    upstream and must raise, not warn.
    """
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError("non-finite values in curve/prototype matrix")
    with np.errstate(all="ignore"):
        return A @ B


def fit_prototypes(curves: np.ndarray, k: int, *, seed: int = 0,
                   n_iter: int = 50) -> np.ndarray:
    """Deterministic k-means++ over shape-normalised curves.

    Returns
    -------
    np.ndarray
        ``[k, n]`` unit-norm prototypes.

    Notes
    -----
    Implemented here rather than via sklearn so the result is reproducible
    across environments (sklearn is an optional dependency in this project and
    its k-means init has changed between versions). Sign is **not** folded:
    ``phi`` and ``-phi`` compose differently downstream and are genuinely
    different learned functions -- U13-C's whole result was about a sign.
    """
    X = np.asarray(curves, dtype=np.float64)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if X.ndim != 2 or X.shape[0] < k:
        raise ValueError(
            f"need at least k={k} curves to fit k prototypes, got shape {X.shape}"
        )
    rng = np.random.default_rng(int(seed))
    n = X.shape[0]

    first = int(rng.integers(n))
    idx = [first]
    d2 = ((X - X[first]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = float(d2.sum())
        cand = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=d2 / total))
        idx.append(cand)
        d2 = np.minimum(d2, ((X - X[cand]) ** 2).sum(axis=1))
    C = X[idx].copy()

    xn = (X ** 2).sum(axis=1)[:, None]
    for _ in range(int(n_iter)):
        dist = xn - 2.0 * _dot(X, C.T) + (C ** 2).sum(axis=1)[None, :]
        lab = dist.argmin(axis=1)
        newC = C.copy()
        for j in range(k):
            m = lab == j
            if m.any():
                newC[j] = X[m].mean(axis=0)
            else:
                # Empty cluster: re-seed on the worst-fit point rather than
                # dropping a prototype, so k stays comparable across arms.
                newC[j] = X[int(dist.min(axis=1).argmax())]
        if np.allclose(newC, C):
            C = newC
            break
        C = newC

    norms = np.linalg.norm(C, axis=1, keepdims=True)
    return C / np.maximum(norms, 1e-12)


def assign_to_prototypes(curves: np.ndarray,
                         prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each (unit-norm) curve to its most similar prototype.

    Returns
    -------
    (labels[E], similarity[E])
        ``similarity`` is the cosine to the assigned prototype, in ``[-1, 1]``.
    """
    X = np.asarray(curves, dtype=np.float64)
    P = np.asarray(prototypes, dtype=np.float64)
    if X.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    sims = _dot(X, P.T)
    labels = sims.argmax(axis=1).astype(np.int64)
    return labels, sims[np.arange(len(labels)), labels]


_SPECTRUM_CAVEAT = (
    "DIAGNOSTIC ONLY. Inside one KANLinear every curve is B @ c over an identical "
    "B-spline basis, so this is a PCA over n_coeff = grid_size + spline_order "
    "coefficients: it cannot exceed n_coeff components and a low-dimensional "
    "spectrum is partly architectural. 'k modes explain X% of variance' is not a "
    "finding; only the trained-vs-untrained margin in mode_verdict is."
)


#: Caveats returned with every result, because each one has already bitten.
CAVEATS: tuple[str, ...] = (
    _SPECTRUM_CAVEAT,
    "ABSOLUTE AGREEMENT SCORES ARE NOT INTERPRETABLE. On synthetic fixtures a "
    "randomly initialised network already scores prototype_fit ~0.63 with k=3, "
    "because 8 spline coefficients over one shared basis cannot produce curves "
    "that are far from every prototype. Quote the trained-vs-untrained margin, "
    "never the trained number on its own.",
    "k IS NOT FREE. Setting k above the number of shapes actually present splits "
    "one true cluster in two and collapses identity_agreement toward chance even "
    "when prototype_fit is ~1.0 (observed on the structured positive control). "
    "Sweep k and report the sweep, or the identity statistic is a k artefact.",
    "Curves are sampled on the shared grid_range, not on each edge's visited "
    "range, so a curve can differ in a region the data never reaches. Each record "
    "carries visited_lo/visited_hi; report coverage alongside any mode claim.",
)


def spectrum_diagnostic(bundle: dict, *, max_components: int = 10) -> dict:
    """PCA spectrum of a normalised bundle -- reported, never claimed.

    Returns
    -------
    dict
        ``{"explained_variance_ratio", "n_coeff", "max_possible_components",
        "n_curves", "is_finding": False, "caveat"}``. The field names are chosen
        so this cannot be quoted as a result: ``is_finding`` is hard-coded
        ``False`` and there is deliberately no ``variance_explained`` key.
    """
    X = np.asarray(bundle["curves"], dtype=np.float64)
    n_coeff = int(bundle.get("n_coeff", 0))
    if X.ndim != 2 or X.shape[0] < 2:
        evr: list[float] = []
    else:
        Xc = X - X.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(Xc, compute_uv=False)
        var = sv ** 2
        total = float(var.sum())
        evr = list((var / total)[:max_components]) if total > 0 else []
    return {
        "explained_variance_ratio": [float(v) for v in evr],
        "n_coeff": n_coeff,
        "max_possible_components": n_coeff,
        "n_curves": int(X.shape[0]) if X.ndim == 2 else 0,
        "is_finding": False,
        "caveat": _SPECTRUM_CAVEAT,
    }


# --------------------------------------------------------------------------- #
# Cross-fold agreement: fit on F-1 folds, assign the held-out one
# --------------------------------------------------------------------------- #

def _modal_label(labels: np.ndarray) -> int:
    return int(np.bincount(labels).argmax())


def _identity_modes(records: Sequence[dict], labels: np.ndarray) -> dict[str, int]:
    """Modal mode per fold-stable identity (``None`` identities are skipped)."""
    by_id: dict[str, list[int]] = {}
    for rec, lab in zip(records, labels):
        ident = rec.get("identity")
        if ident is None:
            continue
        by_id.setdefault(ident, []).append(int(lab))
    return {k: _modal_label(np.asarray(v)) for k, v in by_id.items()}


def cross_fold_agreement(bundles_by_fold: dict, *, k: int = 4, seed: int = 0) -> dict:
    """Leave-one-fold-out prototype agreement for one stratum.

    Fits prototypes on ``F-1`` folds' pooled curves, assigns the held-out fold,
    and scores two things:

    ``prototype_fit``
        mean cosine between a held-out curve and its assigned prototype. Always
        available; this is the primary score because it exists for hidden-fed
        strata too.
    ``identity_agreement``
        fraction of fold-stable identities (e.g. ``kan_z|delay:ytpos``) whose
        modal mode in the held-out fold equals its modal mode across the fitting
        folds. The stronger statement -- "the same physical channel gets the
        same shape in every fold" -- but ``None`` for any stratum whose indices
        are permutation-arbitrary. Chance is ``1/k``.

    Neither number means anything on its own: both are architecture-inflated
    (see :func:`spectrum_diagnostic`). They are inputs to :func:`mode_verdict`,
    which compares them against the untrained null at matched N.

    Returns
    -------
    dict
        JSON-serialisable. ``prototype_fit_mean`` is ``None`` when fewer than
        two folds could be scored -- an unmeasurable score, not a zero.
    """
    folds = list(bundles_by_fold.keys())
    bundles = {f: bundles_by_fold[f] for f in folds}
    if not bundles:
        raise ValueError("cross_fold_agreement needs at least one fold")

    strata = {b.get("stratum") for b in bundles.values()}
    if len(strata) != 1:
        raise ValueError(
            f"cross_fold_agreement refuses to pool more than one stratum: got "
            f"{sorted(map(str, strata))}. Never pool a hidden-fed edge with a "
            "physical-channel edge."
        )
    stratum = strata.pop()
    for b in bundles.values():
        if b.get("gate") is None:
            raise ValueError(
                "cross_fold_agreement expects gated bundles (gate_by_importance "
                "must run on raw importance before any shape normalisation)"
            )
    norm = {f: normalise_curves(b) for f, b in bundles.items()}
    fold_stable = bool(norm[folds[0]].get("fold_stable", False))

    per_fold: dict[str, dict] = {}
    fits: list[float] = []
    agrees: list[float] = []
    n_unmeasurable = 0
    for held in folds:
        train = [norm[f] for f in folds if f != held]
        ho = norm[held]
        train_curves = (np.concatenate([b["curves"] for b in train], axis=0)
                        if train else np.zeros((0, 0)))
        if len(train) == 0 or train_curves.shape[0] < k or ho["n_edges"] < 1:
            n_unmeasurable += 1
            per_fold[str(held)] = {
                "prototype_fit": None, "identity_agreement": None,
                "n_held_out": int(ho["n_edges"]),
                "note": "not enough curves to fit or assign prototypes",
            }
            continue

        protos = fit_prototypes(train_curves, k, seed=seed)
        ho_labels, ho_sims = assign_to_prototypes(ho["curves"], protos)
        fit = float(np.mean(ho_sims))
        fits.append(fit)

        agreement = None
        if fold_stable:
            train_records: list[dict] = []
            train_labels: list[int] = []
            for b in train:
                lab, _ = assign_to_prototypes(b["curves"], protos)
                train_records.extend(b["records"])
                train_labels.extend(lab.tolist())
            train_modes = _identity_modes(train_records, np.asarray(train_labels))
            ho_modes = _identity_modes(ho["records"], ho_labels)
            shared = sorted(set(train_modes) & set(ho_modes))
            if shared:
                agreement = float(
                    np.mean([train_modes[i] == ho_modes[i] for i in shared])
                )
                agrees.append(agreement)
        per_fold[str(held)] = {
            "prototype_fit": fit,
            "identity_agreement": agreement,
            "n_held_out": int(ho["n_edges"]),
            "n_train": int(train_curves.shape[0]),
        }

    pooled = dict(norm[folds[0]])
    pooled["curves"] = np.concatenate([b["curves"] for b in norm.values()], axis=0)
    base_frac = np.concatenate([b["base_fraction"] for b in norm.values()]) \
        if norm else np.zeros(0)

    return {
        "stratum": stratum,
        "fold_stable": fold_stable,
        "k": int(k),
        "n_folds": len(folds),
        "n_folds_unmeasurable": int(n_unmeasurable),
        "per_fold": per_fold,
        "prototype_fit_mean": float(np.mean(fits)) if len(fits) >= 2 else None,
        "prototype_fit_std": float(np.std(fits)) if len(fits) >= 2 else None,
        "identity_agreement_mean": float(np.mean(agrees)) if len(agrees) >= 2 else None,
        "chance_identity_agreement": 1.0 / float(k),
        "base_fraction_mean": float(np.mean(base_frac)) if base_frac.size else None,
        "n_degenerate_dropped": int(sum(b["n_degenerate_dropped"] for b in norm.values())),
        "n_edges_per_fold": {str(f): int(norm[f]["n_edges"]) for f in folds},
        "spectrum_diagnostic": spectrum_diagnostic(pooled),
    }


# --------------------------------------------------------------------------- #
# The untrained null
# --------------------------------------------------------------------------- #

def reinitialised_copy(model, seed: int = 0):
    """Same architecture, same shapes, freshly random weights.

    Deep-copying the *loaded* model and re-initialising is deliberate: it cannot
    drift from the trained model's architecture the way re-constructing from
    guessed constructor args could, and the control is only valid if the two
    networks differ in nothing but their weights. This mirrors
    ``scripts/tsr_control_phase3.py::reinitialised_copy`` (U12-B) including its
    guard; it is duplicated rather than imported because ``src/pri2m`` must not
    depend on ``scripts/``.

    Returns
    -------
    (model, n_reset)
    """
    torch.manual_seed(int(seed))
    fresh = copy.deepcopy(model)
    n_reset = 0
    for module in fresh.modules():
        if module is fresh:
            continue
        if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
            module.reset_parameters()
            n_reset += 1
    if n_reset == 0:
        raise RuntimeError(
            "no submodule exposed reset_parameters(); refusing to run a control "
            "whose 'untrained' network may still carry trained weights"
        )
    fresh.eval()
    return fresh, n_reset


def assert_reinitialised(trained, untrained) -> float:
    """Guard the one way this control can silently become meaningless.

    Returns the mean absolute parameter difference; raises if it is zero (the
    "untrained" arm would be the trained arm) or if no parameters could be
    compared at all.
    """
    diffs, tp = [], dict(trained.named_parameters())
    for name, p in untrained.named_parameters():
        if name in tp and tp[name].shape == p.shape:
            diffs.append(float((tp[name] - p).detach().abs().mean()))
    if not diffs:
        raise RuntimeError("could not compare parameters between the two models")
    mean_diff = float(np.mean(diffs))
    if mean_diff <= 1e-8:
        raise RuntimeError(
            f"re-initialisation did not change the weights (mean |diff| {mean_diff:.2e}); "
            "the 'untrained' condition would be identical to the trained one"
        )
    return mean_diff


def fold_bundles(model, batch: dict, *, n: int = 64, component: str = "spline",
                 strata: Sequence[str] | None = None,
                 max_points: int = 2048) -> dict[str, dict]:
    """One forward pass -> per-stratum curve bundles for one fold/model."""
    stats = collect_activations(model, batch, max_points=max_points)
    return extract_stratified_curves(model, stats, edge_index_map(model),
                                     n=n, component=component, strata=strata)


def untrained_replicate(models_by_fold: dict, batches_by_fold: dict, *, seed: int,
                        **kw) -> tuple[dict, dict]:
    """One complete untrained replicate: every fold re-initialised at ``seed``.

    A replicate mirrors the trained arm's *shape* (the same folds, the same
    batches, the same extraction), so the identical cross-fold statistic can be
    computed on it. One replicate costs one forward pass per fold, which is why
    the API is built to be called many times -- the spread of the null over
    seeds is the denominator of the entire test, and five seeds do not measure a
    spread.

    Returns
    -------
    (bundles_by_stratum_by_fold, diagnostics)
        ``diagnostics`` carries ``n_submodules_reinitialised`` and
        ``mean_weight_diff`` per fold, i.e. proof that the two guards fired.
    """
    out: dict[str, dict] = {}
    diag: dict[str, dict] = {}
    for pos, fold in enumerate(sorted(models_by_fold)):
        model = models_by_fold[fold]
        # Distinct stream per (seed, fold): a shared seed would make every fold
        # of a replicate the same random network.
        fresh, n_reset = reinitialised_copy(model, seed=int(seed) * 1_000_003 + pos)
        mean_diff = assert_reinitialised(model, fresh)
        diag[str(fold)] = {
            "n_submodules_reinitialised": int(n_reset),
            "mean_weight_diff": float(mean_diff),
        }
        for stratum, bundle in fold_bundles(fresh, batches_by_fold[fold], **kw).items():
            out.setdefault(stratum, {})[fold] = bundle
    return out, diag


# --------------------------------------------------------------------------- #
# The pre-committed verdict
# --------------------------------------------------------------------------- #

def mode_verdict(trained_score, untrained_scores, *, threshold: float = 0.8,
                 min_seeds: int = 20, reference: str = "p95") -> dict:
    """Did training produce the recurring shapes, or does random init do this too?

    The rule is fixed **before** any number is seen and mirrors U12-B's
    ``untrained_lock >= 0.8 * |trained_lock| => artifact``
    (``scripts/whitebox_phase3.py::tsr_control_verdict``):

        ``untrained_reference >= threshold * |trained_score| => artifact``

    where ``untrained_reference`` is the 95th percentile of the untrained
    distribution (pre-committed; the mean would understate a heavy-tailed null
    and the max would drift upward with the seed count).

    Parameters
    ----------
    trained_score:
        The trained arm's cross-fold agreement, at matched N. ``None``/NaN means
        unmeasurable.
    untrained_scores:
        One score per untrained seed, at the same matched N.
    threshold:
        Pre-committed margin. Explicit, never implicit.
    min_seeds:
        Refuse to decide on fewer seeds than this. The null's spread is the
        denominator of the test.

    Returns
    -------
    dict
        ``artifact_suspected`` is ``True``, ``False``, or **``None`` = UNDECIDED
        when either side is unmeasurable or the null is under-sampled**. UNDECIDED
        is NOT a pass; ``modes_learned`` is ``None`` in that case too, so a caller
        that reads either field cannot mistake a missing measurement for a
        cleared control.
    """
    scores = [float(s) for s in (untrained_scores or [])
              if s is not None and np.isfinite(float(s))]
    trained_ok = (trained_score is not None
                  and np.isfinite(float(trained_score)))
    stats = {
        "n_seeds": len(scores),
        "mean": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
        "p95": float(np.percentile(scores, 95)) if scores else None,
        "max": float(np.max(scores)) if scores else None,
        "min": float(np.min(scores)) if scores else None,
    }
    base = {
        "trained_score": float(trained_score) if trained_ok else None,
        "untrained": stats,
        "reference": reference,
        "threshold": float(threshold),
        "min_seeds": int(min_seeds),
    }
    if not trained_ok or len(scores) < int(min_seeds):
        why = ("trained score unmeasurable" if not trained_ok
               else f"untrained null under-sampled ({len(scores)} < {min_seeds} seeds)")
        return {**base, "untrained_reference": None, "margin": None, "z": None,
                "artifact_suspected": None, "modes_learned": None,
                "undecided_reason": why}

    ref_value = stats[reference]
    trained = float(trained_score)
    # Compare against |trained|: a trained score at or below zero carries no
    # signal to attribute, so any untrained score at all is disqualifying.
    denom = abs(trained)
    suspected = bool(ref_value >= float(threshold) * denom) if denom > 0 else True
    z = None
    if stats["std"] and stats["std"] > 0:
        z = float((trained - stats["mean"]) / stats["std"])
    return {
        **base,
        "untrained_reference": float(ref_value),
        "margin": float(trained - ref_value),
        "z": z,
        "artifact_suspected": suspected,
        "modes_learned": (not suspected),
        "undecided_reason": None,
    }


# --------------------------------------------------------------------------- #
# Mode ablation edge sets (step 8) -- the controls, built with the target
# --------------------------------------------------------------------------- #

def mode_edge_sets(bundle: dict, labels: np.ndarray, mode: int, *, seed: int = 0,
                   n_trials: int = 200, population: dict | None = None) -> dict:
    """Edge lists for a mode ablation, with its two mandatory controls.

    "Removing 300 edges hurt" says nothing without controls, so this returns
    three same-size sets:

    ``mode_edges``
        every edge assigned to ``mode``.
    ``random_matched``
        drawn from the wider edge population, matched on **count and summed
        importance** (best of ``n_trials`` deterministic draws).
    ``other_modes_matched``
        an equal-size set drawn from the other modes only, disjoint from
        ``mode_edges``.

    Parameters
    ----------
    bundle:
        The gated, normalised bundle the ``labels`` were assigned on.
    labels:
        Mode assignment from :func:`assign_to_prototypes`, one per bundle edge.
    population:
        Wider edge pool for ``random_matched`` -- normally the **ungated**
        bundle, so the random control is drawn from all edges rather than only
        from the ones that already cleared the gate. Defaults to ``bundle``.

    Returns
    -------
    dict
        ``{"mode", "mode_edges", "random_matched", "other_modes_matched",
        "summed_importance", "random_overlap_with_mode", "n_trials"}`` with each
        edge as ``{"kan", "out", "in", "feature", "importance"}`` -- exactly what
        a zero-this-edge routine needs, and JSON-serialisable.

    Notes
    -----
    ``random_matched`` is a genuine random draw, so it *may* contain edges that
    are also in ``mode_edges``; forcing disjointness would make it a
    "random-except-the-interesting-ones" set, which is a different (and weaker)
    control. The overlap is reported as ``random_overlap_with_mode`` rather than
    silently accepted -- a large overlap invalidates that arm's reading.
    """
    labels = np.asarray(labels)
    if labels.shape[0] != bundle["n_edges"]:
        raise ValueError(
            f"labels ({labels.shape[0]}) do not match the bundle "
            f"({bundle['n_edges']}); assign_to_prototypes must run on this bundle"
        )
    rng = np.random.default_rng(int(seed))
    in_mode = np.flatnonzero(labels == int(mode))
    others = np.flatnonzero(labels != int(mode))
    n = int(len(in_mode))

    def _spec(b: dict, i: int) -> dict:
        r = b["records"][int(i)]
        return {"kan": r["kan"], "out": r["out"], "in": r["in"],
                "feature": r["feature"], "importance": r["importance"]}

    target = float(bundle["importance"][in_mode].sum())

    def _best_draw(src_bundle: dict, pool: np.ndarray) -> list[int]:
        """Best-of-n_trials subset of ``pool`` matched on summed importance."""
        if n == 0 or len(pool) == 0:
            return []
        take = min(n, len(pool))
        best, best_err = None, None
        imp = src_bundle["importance"]
        for _ in range(int(n_trials)):
            pick = rng.choice(pool, size=take, replace=False)
            err = abs(float(imp[pick].sum()) - target)
            if best_err is None or err < best_err:
                best, best_err = pick, err
        return [int(i) for i in best]

    pop = population if population is not None else bundle
    pop_pool = np.arange(pop["n_edges"])
    random_idx = _best_draw(pop, pop_pool)
    other_idx = _best_draw(bundle, others)

    return {
        "mode": int(mode),
        "n_edges": n,
        "mode_edges": [_spec(bundle, i) for i in in_mode],
        "random_matched": [_spec(pop, i) for i in random_idx],
        "other_modes_matched": [_spec(bundle, i) for i in other_idx],
        "summed_importance": {
            "mode_edges": target,
            "random_matched": float(pop["importance"][random_idx].sum()) if random_idx else 0.0,
            "other_modes_matched": float(bundle["importance"][other_idx].sum()) if other_idx else 0.0,
        },
        # Sentinel: how much of the "random" control is the mode itself. Only
        # meaningful when population is bundle-relative; reported always.
        "random_overlap_with_mode": int(len(
            {(pop["records"][i]["kan"], pop["records"][i]["out"], pop["records"][i]["in"])
             for i in random_idx}
            & {(bundle["records"][i]["kan"], bundle["records"][i]["out"],
                bundle["records"][i]["in"]) for i in in_mode}
        )),
        "n_trials": int(n_trials),
    }


# --------------------------------------------------------------------------- #
# Top level: one call a thin runner can make
# --------------------------------------------------------------------------- #

def jsonable(obj):
    """Recursively convert to JSON-safe types; non-finite floats become ``None``.

    NaN/inf are mapped to ``None`` on purpose: an unmeasurable value must read
    as *missing*, never as a number that survives a later ``mean()``.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def run_activation_mode_analysis(
    models_by_fold: dict,
    batches_by_fold: dict,
    *,
    n_seeds: int = 32,
    k: int = 4,
    n_points: int = 64,
    component: str = "spline",
    strata: Sequence[str] | None = None,
    gate_quantile: float | None = 0.9,
    gate_min_importance: float | None = None,
    threshold: float = 0.8,
    min_seeds: int = 20,
    score_key: str = "prototype_fit_mean",
    seed: int = 0,
    max_points: int = 2048,
    progress=None,
) -> dict:
    """The whole U14-B measurement: trained arm, untrained null, verdict.

    Parameters
    ----------
    models_by_fold, batches_by_fold:
        ``{fold_id: model}`` and ``{fold_id: batch}``. Loading them is the
        runner's job; this module does no file IO. The untrained arm reuses the
        *same* batches, so the data-aware importance of the two arms is
        comparable.
    n_seeds:
        Untrained replicates. Default 32, and :func:`mode_verdict` refuses to
        decide below ``min_seeds``. Each replicate costs one forward pass per
        fold.
    strata:
        Defaults to :data:`FOLD_STABLE_STRATA`. Pass hidden-fed strata
        explicitly if you want the population-only view; they will report
        ``identity_agreement_mean = None``.
    gate_quantile / gate_min_importance:
        Pass exactly one. ``gate_min_importance`` is the absolute floor that
        makes the arms differ in size -- which is precisely why matched N is
        applied unconditionally afterwards.
    score_key:
        Which agreement statistic feeds the verdict
        (``"prototype_fit_mean"`` or ``"identity_agreement_mean"``).

    Returns
    -------
    dict
        JSON-serialisable. Per stratum: the trained agreement, the untrained
        distribution, the matched-N record, the gate pass fractions of both
        arms, the base fraction, the spectrum **diagnostic**, and the verdict
        (``artifact_suspected`` True/False/**None = UNDECIDED, never a pass**).
        Plus a ``headline`` string that states the verdict in words.
    """
    if (gate_quantile is None) == (gate_min_importance is None):
        raise ValueError("pass exactly one of gate_quantile= or gate_min_importance=")
    want = tuple(sorted(FOLD_STABLE_STRATA)) if strata is None else tuple(strata)
    folds = sorted(models_by_fold)
    if set(folds) != set(batches_by_fold):
        raise ValueError("models_by_fold and batches_by_fold must cover the same folds")

    extract_kw = dict(n=n_points, component=component, strata=want,
                      max_points=max_points)

    def _tick(stage: str, done: int, total: int) -> None:
        if progress is not None:
            progress(stage, done, total)

    # ---- trained arm ------------------------------------------------------ #
    trained: dict[str, dict] = {}
    for pos, fold in enumerate(folds):
        for stratum, bundle in fold_bundles(models_by_fold[fold],
                                            batches_by_fold[fold],
                                            **extract_kw).items():
            trained.setdefault(stratum, {})[fold] = bundle
        _tick("trained", pos + 1, len(folds))

    # ---- untrained null --------------------------------------------------- #
    replicates: list[dict] = []
    guards: list[dict] = []
    for s in range(int(n_seeds)):
        bundles, diag = untrained_replicate(models_by_fold, batches_by_fold,
                                            seed=int(seed) * 7919 + s, **extract_kw)
        replicates.append(bundles)
        guards.append(diag)
        _tick("untrained", s + 1, int(n_seeds))

    gate_kw = ({"quantile": gate_quantile} if gate_quantile is not None
               else {"min_importance": gate_min_importance})

    out_strata: dict[str, dict] = {}
    for stratum in want:
        if stratum not in trained:
            out_strata[stratum] = {
                "trained": None, "untrained": None, "matched_n": {}, "gate": None,
                "spectrum_diagnostic": None,
                "verdict": mode_verdict(None, [], threshold=threshold,
                                        min_seeds=min_seeds),
                "note": "stratum not present in this model",
            }
            continue

        # Gate FIRST, on raw data-aware importance, per arm.
        t_gated = {f: gate_by_importance(b, **gate_kw)
                   for f, b in trained[stratum].items()}
        u_gated = [
            {f: gate_by_importance(b, **gate_kw)
             for f, b in rep.get(stratum, {}).items()}
            for rep in replicates
        ]

        # Snapshot the pre-match gate record. If the trained arm passes 90% and
        # the untrained arm 5%, the two were not the same population and matched
        # N is doing real work -- the reader must be able to see that, and
        # match_gate_sizes overwrites the counts.
        gate_report = {
            "mode": t_gated[folds[0]]["gate"]["mode"],
            "param": t_gated[folds[0]]["gate"]["param"],
            "trained_pass_fraction": {str(f): t_gated[f]["gate"]["pass_fraction"]
                                      for f in folds},
            "trained_threshold": {str(f): t_gated[f]["gate"]["threshold"]
                                  for f in folds},
            "trained_importance_median": {str(f): t_gated[f]["gate"]["importance_median"]
                                          for f in folds},
            "untrained_pass_fraction_mean": _mean_or_none(
                [u[f]["gate"]["pass_fraction"] for u in u_gated for f in u]),
            "untrained_importance_median_mean": _mean_or_none(
                [u[f]["gate"]["importance_median"] for u in u_gated for f in u]),
        }

        # THEN match N per fold across every arm, so the trained and untrained
        # populations are the same size before any statistic is compared.
        matched_n: dict[str, int] = {}
        for f in folds:
            arms = [t_gated[f]] + [u[f] for u in u_gated if f in u]
            fixed = match_gate_sizes(arms)
            t_gated[f] = fixed[0]
            for u, b in zip([u for u in u_gated if f in u], fixed[1:]):
                u[f] = b
            matched_n[str(f)] = int(fixed[0]["n_edges"])

        trained_res = cross_fold_agreement(t_gated, k=k, seed=seed)
        untrained_res = [cross_fold_agreement(u, k=k, seed=seed)
                         for u in u_gated if len(u) >= 2]
        untrained_scores = [r[score_key] for r in untrained_res]
        verdict = mode_verdict(trained_res[score_key], untrained_scores,
                               threshold=threshold, min_seeds=min_seeds)

        u_fit = [r["prototype_fit_mean"] for r in untrained_res
                 if r["prototype_fit_mean"] is not None]
        out_strata[stratum] = {
            "trained": trained_res,
            "untrained": {
                "n_seeds": len(untrained_res),
                "score_key": score_key,
                "scores": untrained_scores,
                "mean": float(np.mean(u_fit)) if u_fit else None,
                "std": float(np.std(u_fit)) if u_fit else None,
                "p95": float(np.percentile(u_fit, 95)) if u_fit else None,
                "max": float(np.max(u_fit)) if u_fit else None,
                "identity_agreement_mean": _mean_or_none(
                    [r["identity_agreement_mean"] for r in untrained_res]),
                "base_fraction_mean": _mean_or_none(
                    [r["base_fraction_mean"] for r in untrained_res]),
                "spectrum_diagnostic": (untrained_res[0]["spectrum_diagnostic"]
                                        if untrained_res else None),
            },
            "matched_n": matched_n,
            "gate": gate_report,
            "verdict": verdict,
            "spectrum_diagnostic": trained_res["spectrum_diagnostic"],
        }

    return jsonable({
        "config": {
            "n_seeds": int(n_seeds), "k": int(k), "n_points": int(n_points),
            "component": component, "strata": list(want),
            "gate_quantile": gate_quantile,
            "gate_min_importance": gate_min_importance,
            "threshold": float(threshold), "min_seeds": int(min_seeds),
            "score_key": score_key, "seed": int(seed),
            "folds": [str(f) for f in folds],
        },
        # Proof that both re-initialisation guards fired on every replicate.
        # The MINIMUM weight difference is the sentinel: an average would hide a
        # single replicate that came back identical to the trained model.
        "null_guards": {
            "n_replicates": len(guards),
            "min_mean_weight_diff": min(
                (d["mean_weight_diff"] for g in guards for d in g.values()),
                default=None),
            "example": guards[0] if guards else None,
        },
        "strata": out_strata,
        "caveats": list(CAVEATS),
        "headline": _headline(out_strata),
    })


def _mean_or_none(values) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _headline(out_strata: dict) -> str:
    """One line per stratum, stating the verdict in words.

    UNDECIDED is spelled out rather than left as a bare ``None`` -- the whole
    point of this module is that a missing measurement must never be skimmed as
    a pass.
    """
    parts = []
    for stratum, res in sorted(out_strata.items()):
        v = res.get("verdict", {})
        a = v.get("artifact_suspected")
        if a is None:
            parts.append(f"{stratum}: UNDECIDED ({v.get('undecided_reason')}) "
                         "-- this is NOT a pass")
        elif a:
            parts.append(
                f"{stratum}: ARTIFACT SUSPECTED -- randomly initialised networks "
                f"reach {v.get('untrained_reference')} against the trained "
                f"{v.get('trained_score')}; the modes are basis geometry, not "
                "learned structure"
            )
        else:
            parts.append(
                f"{stratum}: MODES SURVIVE THE UNTRAINED NULL "
                f"(trained {v.get('trained_score')} vs untrained p95 "
                f"{v.get('untrained_reference')}, margin {v.get('margin')})"
            )
    return " | ".join(parts) if parts else "no strata measured -- UNDECIDED"
