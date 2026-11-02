"""Decision rule of the flag ablation (scripts/ablation_flag.py)."""
import pytest

from scripts.ablation_flag import conclusion


@pytest.mark.parametrize("recovered, expected", [
    (0.97, "mainly the availability pattern"), (0.7, "mainly the availability pattern"),
    (0.5, "mixed"), (0.3, "mainly the content of the fields"), (-0.1, "mainly the content of the fields"),
])
def test_conclusion(recovered, expected):
    assert conclusion(recovered) == expected
