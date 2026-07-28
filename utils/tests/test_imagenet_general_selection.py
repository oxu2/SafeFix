from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.imagenet_general_selection as selection


def _bundle(labels: np.ndarray, attributes: dict[str, np.ndarray]) -> SimpleNamespace:
    size = len(labels)
    return SimpleNamespace(
        train_indices=np.arange(size, dtype=np.int64),
        val_indices=np.empty(0, dtype=np.int64),
        test_indices=np.empty(0, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        train_labels=np.asarray(labels, dtype=np.int64),
        attributes=attributes,
        class_names=[str(value) for value in sorted(np.unique(labels))],
    )


def test_stratified_sample_is_balanced_and_reproducible() -> None:
    labels = np.repeat(np.arange(4), [10, 9, 8, 7])
    first = selection.stratified_sample(labels, n_select=13, seed=42)
    second = selection.stratified_sample(labels, n_select=13, seed=42)

    assert first == second
    assert len(first) == len(set(first)) == 13
    counts = np.bincount(labels[first], minlength=4)
    assert counts.tolist() == [4, 3, 3, 3]


def test_classwise_coreset_respects_class_quota() -> None:
    train_labels = np.repeat(np.arange(3), 5)
    reference_labels = np.repeat(np.arange(3), 2)
    train_vectors = np.eye(15, dtype=np.float32)
    reference_vectors = np.zeros((6, 15), dtype=np.float32)
    for row, label in enumerate(reference_labels):
        reference_vectors[row, label * 5 + row % 2] = 1.0

    selected = selection.classwise_coreset(
        train_vectors,
        train_labels,
        reference_vectors,
        reference_labels,
        n_select=9,
        seed=7,
    )

    assert len(selected) == len(set(selected)) == 9
    assert np.bincount(train_labels[selected], minlength=3).tolist() == [3, 3, 3]


def test_object_attribute_entropy_reduction_prioritizes_under_labeled_value() -> None:
    bundle = _bundle(
        labels=np.asarray([0, 0, 0, 1, 1]),
        attributes={"color": np.asarray(["red", "red", "red", "red", "blue"], dtype=object)},
    )
    probabilities = np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)

    scores = selection.object_attribute_entropy_reduction_scores(
        bundle=bundle,
        labeled_local_indices=np.asarray([0, 1, 2]),
        candidate_local_indices=np.asarray([3, 4]),
        candidate_probabilities=probabilities,
        attribute_config=selection.AttributeConfig(fields=("color",)),
    )

    assert scores[1] > scores[0]


def test_attribute_query_does_not_use_unlabeled_ground_truth() -> None:
    labels = np.asarray([0, 1, 0, 0, 1, 1], dtype=np.int64)
    attributes = {
        "color": np.asarray(["red", "blue", "green", "yellow", "black", "white"], dtype=object)
    }
    bundle = _bundle(labels, attributes)
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.55, 0.45],
            [0.51, 0.49],
            [0.45, 0.55],
            [0.49, 0.51],
        ],
        dtype=np.float64,
    )
    embeddings = np.arange(18, dtype=np.float32).reshape(6, 3)
    with patch.object(selection, "split_dataset", lambda *args, **kwargs: object()), patch.object(
        selection,
        "model_outputs",
        lambda *args, **kwargs: (probabilities, embeddings),
    ):
        kwargs = dict(
            method="attriselect_al",
            model=object(),
            labeled=np.asarray([0, 1], dtype=np.int64),
            unlabeled=np.asarray([2, 3, 4, 5], dtype=np.int64),
            query_size=2,
            seed=42,
            training_config=selection.TrainingConfig(workers=0),
            attribute_config=selection.AttributeConfig(fields=("color",)),
            device=torch.device("cpu"),
        )
        selected_before, _ = selection._active_query(bundle=bundle, **kwargs)

        bundle.labels[2:] = 1 - bundle.labels[2:]
        bundle.train_labels[2:] = bundle.labels[2:]
        selected_after, _ = selection._active_query(bundle=bundle, **kwargs)

    assert selected_before.tolist() == selected_after.tolist()


def test_active_learning_round_sizes_and_query_uniqueness() -> None:
    labels = np.repeat(np.arange(2), 10)
    attributes = {"color": np.asarray([f"value-{index % 4}" for index in range(20)], dtype=object)}
    bundle = _bundle(labels, attributes)

    def fake_train(*args, **kwargs):
        return object(), {
            "selected_count": len(args[1]),
            "best_epoch": 1,
            "best_val_accuracy": 0.0,
            "test_accuracy": 0.0,
            "test_loss": 0.0,
            "training_seconds": 0.0,
            "history": [],
            "test_predictions": np.empty(0, dtype=np.int64),
            "test_labels": np.empty(0, dtype=np.int64),
        }

    def fake_query(*, unlabeled, query_size, **kwargs):
        return unlabeled[:query_size], {"selection_seconds": 0.0}

    with patch.object(selection, "train_fresh_model", fake_train), patch.object(
        selection, "_active_query", fake_query
    ), patch.object(selection, "attribute_test_metrics", lambda *args, **kwargs: {}):
        result, manifest = selection.run_active_learning_experiment(
            bundle=bundle,
            method="random",
            seed=42,
            rounds=3,
            initial_ratio=0.2,
            query_ratio=0.2,
            training_config=selection.TrainingConfig(epochs=1, workers=0),
            attribute_config=selection.AttributeConfig(fields=("color",)),
            device=torch.device("cpu"),
        )

    assert [row["labeled_count"] for row in result["round_results"]] == [4, 8, 12]
    assert result["final_labeled_count"] == 12
    initial = set(manifest["initial_local_indices"])
    first_query = set(manifest["round_queries"][0]["queried_local_indices"])
    second_query = set(manifest["round_queries"][1]["queried_local_indices"])
    assert initial.isdisjoint(first_query)
    assert initial.isdisjoint(second_query)
    assert first_query.isdisjoint(second_query)


if __name__ == "__main__":
    tests = [
        test_stratified_sample_is_balanced_and_reproducible,
        test_classwise_coreset_respects_class_quota,
        test_object_attribute_entropy_reduction_prioritizes_under_labeled_value,
        test_attribute_query_does_not_use_unlabeled_ground_truth,
        test_active_learning_round_sizes_and_query_uniqueness,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
