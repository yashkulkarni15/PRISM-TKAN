#!/usr/bin/env python3
"""Render the Phase 3 TKAN white-box report from a run manifest.

Reads ``whitebox_manifest.json`` (produced by ``scripts/whitebox_phase3.py``)
and fills the ``docs/phase3_tkan/whitebox_analysis.md`` TEMPLATE by substituting
every ``{{placeholder}}`` with the corresponding value from the manifest's
``report`` block. Any placeholder the manifest does NOT supply is written as
``N/A (not run)`` -- so the rendered report contains only real, computed values
or an explicit "not run", never an invented number.

Example::

    python scripts/render_whitebox_report.py \
        --manifest results/phase3/xai/fold2/whitebox_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TEMPLATE = _REPO_ROOT / "docs" / "phase3_tkan" / "whitebox_analysis.md"

_MISSING = "N/A (not run)"
_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
_LEADING_COMMENT = re.compile(r"^\s*<!--.*?-->\s*", re.DOTALL)


def _format(value) -> str:
    """Compact string form: floats to ~4 significant figures, everything else str()."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_report(template_text: str, values: dict, missing: str = _MISSING) -> str:
    """Substitute ``{{key}}`` placeholders in ``template_text`` from ``values``.

    The leading HTML comment block (the template's placeholder inventory) is
    stripped. Unknown placeholders become ``missing`` -- never a guess.

    Parameters
    ----------
    template_text:
        The report template (markdown with ``{{key}}`` placeholders).
    values:
        Flat mapping of placeholder name -> value (str / number / bool).
    missing:
        Text substituted for any placeholder absent from ``values``.

    Returns
    -------
    str
        The rendered markdown with every placeholder resolved.
    """
    text = _LEADING_COMMENT.sub("", template_text, count=1)

    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key in values and values[key] is not None:
            return _format(values[key])
        return missing

    return _PLACEHOLDER.sub(repl, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to whitebox_manifest.json.")
    parser.add_argument(
        "--template", default=str(_DEFAULT_TEMPLATE),
        help="Report template (default: docs/phase3_tkan/whitebox_analysis.md).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output path (default: <manifest dir>/whitebox_analysis.md).",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    # The runner nests render-ready values under "report"; fall back to the
    # whole manifest so a flat manifest still renders.
    values = manifest.get("report", manifest) if isinstance(manifest, dict) else {}

    template = Path(args.template).read_text(encoding="utf-8")
    rendered = render_report(template, values)

    out_path = Path(args.out) if args.out else Path(args.manifest).parent / "whitebox_analysis.md"
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Rendered white-box report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
