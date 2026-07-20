"""Semantic naming for every ``KANLinear`` input edge of a Phase 3 TKAN model.

The TKAN encoder feeds each gate ``g = cat([x_t, h_prev, x_delay])`` (layer 0,
delay-aware) so a raw driver index like ``in[69]`` is opaque. :func:`edge_index_map`
turns each input index into ``{index, block, name}`` -- e.g. ``in[69]`` becomes
``block="x_delay", name="delay:ytpos"`` -- so the white-box report can state *which*
physical quantity (a current channel, a hidden unit, or the tau-delayed channel that
carries the regenerative/DDE chatter term) an edge reads.

The map is structural (no forward pass) and covers every ``KANLinear`` in the model:
encoder gates and severity heads get rich semantics; any other ``KANLinear`` gets a
generic ``block="input", name="in{i}"`` fallback so downstream lookups never miss.
"""

from __future__ import annotations

from collections.abc import Sequence

from pri2m.models.kan import KANLinear

DEFAULT_CHANNELS: tuple[str, ...] = ("xtpos", "ytpos", "Forcex", "Forcey")


def _channel_names(n: int, channel_names: Sequence[str]) -> list[str]:
    """Names for an ``n``-wide input-channel block (named channels if shapes match)."""
    if n == len(channel_names):
        return list(channel_names)
    return [f"ch{i}" for i in range(n)]


def _encoder_gate_map(
    layer_idx: int, gate: str, cell, channel_names: Sequence[str]
) -> list[dict]:
    """Index map for one encoder gate ``kan_z|kan_r|kan_h`` of one cell.

    Block layout mirrors :meth:`TKANCell.forward`'s gate input
    ``cat([x_t, h_prev (or r*h_prev for kan_h), x_delay?])``.
    """
    in_size = cell.input_size
    hid = cell.hidden_size
    entries: list[dict] = []

    # x_t block: raw channels on the bottom (delay-aware) layer, else the
    # previous layer's hidden state.
    if layer_idx == 0:
        xt_names = _channel_names(in_size, channel_names)
    else:
        xt_names = [f"L{layer_idx - 1}_h{i}" for i in range(in_size)]
    for i, name in enumerate(xt_names):
        entries.append({"index": len(entries), "block": "x_t", "name": name})

    # h_prev block (reset-gated for the candidate gate kan_h).
    hblock = "r*h_prev" if gate == "kan_h" else "h_prev"
    for j in range(hid):
        entries.append({"index": len(entries), "block": hblock, "name": f"h{j}"})

    # x_delay block: only present when the cell is delay-aware.
    if cell.use_delay:
        delay_names = _channel_names(in_size, channel_names)
        for name in delay_names:
            entries.append(
                {"index": len(entries), "block": "x_delay", "name": f"delay:{name}"}
            )
    return entries


def _severity_head_map(head: KANLinear, hidden_size: int, side_input: bool) -> list[dict]:
    """Index map for a severity ``reg_head``/``cls_head`` (hidden | side blocks)."""
    width = head.in_features
    n_side = max(0, width - hidden_size)
    entries: list[dict] = []
    for j in range(min(hidden_size, width)):
        entries.append({"index": j, "block": "hidden", "name": f"h{j}"})
    if side_input and n_side > 0:
        for m in range(n_side):
            entries.append(
                {"index": hidden_size + m, "block": "side", "name": f"side{m}"}
            )
    return entries


def edge_index_map(model, channel_names: Sequence[str] = DEFAULT_CHANNELS) -> dict:
    """Map every ``KANLinear`` input edge of ``model`` to ``{index, block, name}``.

    Parameters
    ----------
    model:
        A :class:`~pri2m.models.tkan_model.TKANModel` (or any module containing
        ``KANLinear`` layers; unknown layers get a generic fallback map).
    channel_names:
        Names for the raw input channels (default
        ``("xtpos", "ytpos", "Forcex", "Forcey")``).

    Returns
    -------
    dict
        ``{qualified_kanlinear_name: [ {"index", "block", "name"}, ... ]}`` with
        one list entry per input edge, in index order. Covers every ``KANLinear``
        in ``model``.
    """
    out: dict[str, list[dict]] = {}

    # Encoder gates.
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "layers"):
        for layer_idx, cell in enumerate(encoder.layers):
            for gate in ("kan_z", "kan_r", "kan_h"):
                module = getattr(cell, gate, None)
                if isinstance(module, KANLinear):
                    name = f"encoder.layers.{layer_idx}.{gate}"
                    out[name] = _encoder_gate_map(layer_idx, gate, cell, channel_names)

    # Severity heads.
    severity = getattr(model, "severity", None)
    hidden_size = int(getattr(model, "hidden_size", 0))
    side_input = bool(getattr(model, "side_input", False))
    if severity is not None:
        for head_name in ("reg_head", "cls_head"):
            head = getattr(severity, head_name, None)
            if isinstance(head, KANLinear):
                out[f"severity.{head_name}"] = _severity_head_map(
                    head, hidden_size, side_input
                )

    # Fallback: any other KANLinear (e.g. a gated trajectory head) gets a
    # generic map so downstream name lookups never miss.
    for qualname, module in model.named_modules():
        if isinstance(module, KANLinear) and qualname not in out:
            out[qualname] = [
                {"index": i, "block": "input", "name": f"in{i}"}
                for i in range(module.in_features)
            ]
    return out
