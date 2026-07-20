"""Cross-fold overlay for fuzz-manifest replication runs."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))


def _manifest(tmp_path, name, shift):
    levels = [0.0, 0.5, 1.0]
    manifest = {"fuzzers": {"input_noise": {
        "levels": levels,
        "results": [
            {
                "level": level,
                "mean": {"auc": 0.95 - shift * i, "delay_drop": 0.3 - 0.1 * i},
                "sd": {"auc": 0.01, "delay_drop": 0.01},
            }
            for i, level in enumerate(levels)
        ],
        "breakdown": {
            "auc_below_075": None,
            "drop_below_005": None,
            "auc_at_chance": None,
        },
    }}}
    path = tmp_path / name
    path.write_text(json.dumps(manifest))
    return path


def test_replication_overlay(tmp_path):
    import plot_fuzz_replication as replication

    m1 = _manifest(tmp_path, "a.json", 0.05)
    m2 = _manifest(tmp_path, "b.json", 0.10)
    out = tmp_path / "rep.png"
    rc = replication.main([
        "--manifests", str(m1), str(m2),
        "--labels", "fold0", "fold1",
        "--out", str(out),
    ])
    assert rc == 0 and out.is_file()
