import pytest

from utils.training_milestones import parse_evaluation_milestones


def test_parse_evaluation_milestones_sorts_and_deduplicates():
    assert parse_evaluation_milestones("30,15,30,60", num_epochs=60) == [15, 30, 60]


def test_parse_evaluation_milestones_accepts_empty_value():
    assert parse_evaluation_milestones(None, num_epochs=60) == []
    assert parse_evaluation_milestones("", num_epochs=60) == []


@pytest.mark.parametrize("value", ["zero", "1.5", "15,bad"])
def test_parse_evaluation_milestones_rejects_non_integers(value):
    with pytest.raises(ValueError, match="must be integers"):
        parse_evaluation_milestones(value, num_epochs=60)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_evaluation_milestones_rejects_non_positive_values(value):
    with pytest.raises(ValueError, match="must be positive"):
        parse_evaluation_milestones(value, num_epochs=60)


def test_parse_evaluation_milestones_rejects_epoch_beyond_training():
    with pytest.raises(ValueError, match="exceeds --num_epochs"):
        parse_evaluation_milestones("15,61", num_epochs=60)
