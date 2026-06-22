import pytest
from pri2m.phase3.splits import parent_group


def _arith(d):
    if 91 <= d <= 120:
        return d - 90
    if 61 <= d <= 90:
        return d - 60
    return d


@pytest.mark.parametrize("d,expected", [
    (23, 1), (83, 1), (113, 1),       # 23-cluster merges into 1
    (17, 16), (77, 16), (107, 16),    # 17-cluster merges into 16
    (139, 14), (121, 14),             # new clean 139 + new noisy 121 -> clean 14
    (126, 1), (148, 1),               # new clean repeats of 1
    (150, 25), (155, 30),             # new Cn clean repeats of 25 / 30
    (122, 122), (123, 123), (124, 124), (125, 125),  # novel noisy -> own group
])
def test_known_mappings(d, expected):
    assert parent_group(d) == expected


def test_reproduces_arithmetic_except_known_twins():
    twins = {17, 23, 77, 83, 107, 113}
    for d in range(1, 121):
        if d in twins:
            continue
        assert parent_group(d) == _arith(d), f"changed non-twin {d}"


def test_only_merges_never_splits():
    for d in range(1, 121):
        assert parent_group(d) == parent_group(_arith(d))
