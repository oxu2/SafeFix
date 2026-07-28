import pytest

from scripts.summarize_imagenet_fr_epoch_multiplier_rescaled import (
    compute_fix_rate,
    validate_complete,
)


def test_compute_fix_rate_matches_manuscript_definition():
    assert compute_fix_rate(0.7562, 0.7173) == pytest.approx(13.76017, abs=1e-5)


@pytest.mark.parametrize("before", [-0.1, 1.0, 1.1])
def test_compute_fix_rate_rejects_invalid_before_accuracy(before):
    with pytest.raises(ValueError, match="Before accuracy"):
        compute_fix_rate(0.75, before)


def test_validate_complete_accepts_attriselect_only_grid():
    records = [
        {"strategy": "attribute", "epochs": epochs}
        for epochs in (15, 30, 45, 60)
    ]
    validate_complete(records, ("attribute",))
