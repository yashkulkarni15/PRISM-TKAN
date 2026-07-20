"""xai.semantics.edge_index_map: name every KANLinear input edge.

The encoder gate input is ``cat([x_t, h_prev, x_delay])`` and the trained model's
top drivers are raw indices like ``in[69]``. This map turns those indices into
``current Forcex`` / ``hidden h12`` / ``DELAYED Forcey`` so the white-box report
can say *which* physical quantity an edge reads.
"""

from __future__ import annotations

from pri2m.models.tkan_model import TKANModel
from pri2m.xai.semantics import edge_index_map

CH = ("xtpos", "ytpos", "Forcex", "Forcey")


def test_layer0_gate_map_blocks_and_names():
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    m = edge_index_map(model, channel_names=CH)

    z = m["encoder.layers.0.kan_z"]
    assert len(z) == 4 + 8 + 4  # x_t + h_prev + x_delay
    # x_t block -> current channels
    assert [e["block"] for e in z[:4]] == ["x_t"] * 4
    assert [e["name"] for e in z[:4]] == list(CH)
    # h_prev block
    assert [e["block"] for e in z[4:12]] == ["h_prev"] * 8
    assert z[4]["name"] == "h0" and z[11]["name"] == "h7"
    # x_delay block -> DELAYED channels (the regenerative/DDE term)
    assert [e["block"] for e in z[12:16]] == ["x_delay"] * 4
    assert z[12]["name"] == "delay:xtpos"
    assert z[14]["name"] == "delay:Forcex"
    # indices are 0..in_features-1 in order
    assert [e["index"] for e in z] == list(range(16))


def test_candidate_gate_uses_reset_gated_hidden_block():
    """kan_h's middle block is r*h_prev (reset-gated), labelled distinctly."""
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    m = edge_index_map(model, channel_names=CH)
    h = m["encoder.layers.0.kan_h"]
    assert [e["block"] for e in h[4:12]] == ["r*h_prev"] * 8


def test_severity_head_hidden_vs_side_blocks():
    model = TKANModel(
        input_size=4, hidden_size=8, num_layers=1,
        side_input=True, side_features=("omega", "tpf", "tau"), side_hidden=4,
    )
    m = edge_index_map(model, channel_names=CH)
    reg = m["severity.reg_head"]
    assert len(reg) == 8 + 4  # hidden + side_hidden
    assert [e["block"] for e in reg[:8]] == ["hidden"] * 8
    assert [e["block"] for e in reg[8:12]] == ["side"] * 4
    assert reg[8]["name"] == "side0"


def test_severity_head_no_side_is_all_hidden():
    model = TKANModel(input_size=4, hidden_size=8, num_layers=1, side_input=False)
    m = edge_index_map(model, channel_names=CH)
    reg = m["severity.reg_head"]
    assert len(reg) == 8
    assert all(e["block"] == "hidden" for e in reg)


def test_upper_layer_has_no_delay_block():
    """Layer >0 consumes the previous layer's hidden states; no delay block."""
    model = TKANModel(input_size=4, hidden_size=8, num_layers=2, side_input=False)
    m = edge_index_map(model, channel_names=CH)
    z1 = m["encoder.layers.1.kan_z"]
    assert len(z1) == 8 + 8  # x_t(=prev hidden) + h_prev, no delay
    assert [e["block"] for e in z1[:8]] == ["x_t"] * 8
    assert z1[0]["name"] == "L0_h0"  # previous-layer hidden
    assert [e["block"] for e in z1[8:16]] == ["h_prev"] * 8
    assert not any(e["block"] == "x_delay" for e in z1)
