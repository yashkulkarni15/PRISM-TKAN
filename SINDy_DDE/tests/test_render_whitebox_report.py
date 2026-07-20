"""render_whitebox_report: fill the white-box report TEMPLATE from a manifest.

The renderer is deliberately dumb: it substitutes ``{{key}}`` placeholders with
values the runner already computed, and writes ``N/A (not run)`` for anything the
manifest does not supply -- so the rendered report NEVER contains an invented
number, only real values or an explicit "not run".
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import render_whitebox_report as R  # noqa: E402


def test_substitutes_known_and_marks_missing():
    template = "# {{title}}\nAUC is {{val_auc}}\nMissing: {{nope}}\n"
    out = R.render_report(template, {"title": "Report", "val_auc": 0.955})
    assert "# Report" in out
    assert "0.955" in out
    assert "N/A (not run)" in out
    assert "{{" not in out  # every placeholder resolved


def test_strips_leading_html_comment():
    template = "<!-- inventory: {{a}} {{b}} -->\nBody {{a}}\n"
    out = R.render_report(template, {"a": "X"})
    assert "inventory" not in out  # comment block removed
    assert "Body X" in out


def test_floats_formatted_compactly():
    out = R.render_report("v={{x}}", {"x": 0.12345678})
    # rounded, not the full float repr
    assert "0.123" in out
    assert "0.12345678" not in out


def test_renders_real_template_without_leftover_placeholders(tmp_path):
    template_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "docs" / "phase3_tkan" / "whitebox_analysis.md"
    )
    template = template_path.read_text(encoding="utf-8")
    # Supply only a couple of keys; everything else must degrade to N/A.
    out = R.render_report(template, {"checkpoint": "best_auc.pt", "val_auc": 0.975})
    assert "best_auc.pt" in out
    assert "{{" not in out and "}}" not in out  # no unresolved placeholders
    assert "N/A (not run)" in out  # unsupplied fields marked, not invented


def test_main_writes_file(tmp_path):
    import json

    template_path = tmp_path / "tmpl.md"
    template_path.write_text("# {{title}}\n{{val_auc}}\n", encoding="utf-8")
    manifest_path = tmp_path / "whitebox_manifest.json"
    manifest_path.write_text(
        json.dumps({"report": {"title": "T", "val_auc": 0.9}}), encoding="utf-8"
    )
    out_path = tmp_path / "rendered.md"
    rc = R.main([
        "--manifest", str(manifest_path),
        "--template", str(template_path),
        "--out", str(out_path),
    ])
    assert rc == 0
    text = out_path.read_text(encoding="utf-8")
    assert "# T" in text and "0.9" in text
