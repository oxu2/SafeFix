import unittest

from utils.attribute_aware_baselines import (
    ATTRIBUTE_FILTERED_ENTROPY,
    ATTRIBUTE_FILTERED_RANDOM,
    TARGET_SLICE_STRATIFIED,
    build_attribute_match_pools,
    select_attribute_aware_baseline,
)


class AttributeAwareBaselineTest(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "color": {
                "data": ["red", "red", "blue", "red", "green", "blue", "red", "black"]
            },
            "weather": {
                "data": ["cloudy", "sunny", "cloudy", "cloudy", "clear", "sunny", "clear", "cloudy"]
            },
        }
        self.metadata_indices = list(range(8))
        self.class_labels = [0, 0, 0, 1, 1, 1, 0, 1]
        self.generated_counts = {
            "color": {"red": 1000},
            "weather": {"cloudy": 1000},
        }

    def test_match_pools_use_exact_conjunction_and_partial_union(self):
        exact, partial, background = build_attribute_match_pools(
            attribute_metadata=self.metadata,
            metadata_indices=self.metadata_indices,
            target_values={"color": ("red",), "weather": ("cloudy",)},
        )
        self.assertEqual(exact, [0, 3])
        self.assertEqual(partial, [0, 1, 2, 3, 6, 7])
        self.assertEqual(background, [4, 5])

    def test_target_slice_baseline_prioritizes_all_exact_examples(self):
        selected, diagnostics = select_attribute_aware_baseline(
            strategy=TARGET_SLICE_STRATIFIED,
            attribute_metadata=self.metadata,
            metadata_indices=self.metadata_indices,
            class_labels=self.class_labels,
            generated_attribute_counts=self.generated_counts,
            n_select=5,
            seed=42,
        )
        self.assertEqual(len(selected), 5)
        self.assertTrue({0, 3}.issubset(selected))
        self.assertEqual(diagnostics["selected_exact_count"], 2)

    def test_filtered_random_is_deterministic_and_stays_in_partial_pool(self):
        kwargs = dict(
            strategy=ATTRIBUTE_FILTERED_RANDOM,
            attribute_metadata=self.metadata,
            metadata_indices=self.metadata_indices,
            class_labels=self.class_labels,
            generated_attribute_counts=self.generated_counts,
            n_select=4,
            seed=1234,
        )
        first, first_diagnostics = select_attribute_aware_baseline(**kwargs)
        second, second_diagnostics = select_attribute_aware_baseline(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertEqual(first_diagnostics["selected_partial_count"], 4)

    def test_filtered_entropy_uses_entropy_inside_same_partial_pool(self):
        scores = [0.1, 0.9, 0.8, 0.2, 1.0, 0.7, 0.6, 0.5]
        selected, diagnostics = select_attribute_aware_baseline(
            strategy=ATTRIBUTE_FILTERED_ENTROPY,
            attribute_metadata=self.metadata,
            metadata_indices=self.metadata_indices,
            class_labels=[0] * 8,
            generated_attribute_counts=self.generated_counts,
            n_select=3,
            seed=42,
            entropy_scores=scores,
        )
        self.assertEqual(selected, [1, 2, 6])
        self.assertEqual(diagnostics["selected_partial_count"], 3)

    def test_filtered_pool_shortage_fills_from_background(self):
        selected, diagnostics = select_attribute_aware_baseline(
            strategy=ATTRIBUTE_FILTERED_RANDOM,
            attribute_metadata=self.metadata,
            metadata_indices=self.metadata_indices,
            class_labels=self.class_labels,
            generated_attribute_counts=self.generated_counts,
            n_select=8,
            seed=42,
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(set(selected)), 8)
        self.assertEqual(diagnostics["selected_partial_count"], 6)

    def test_missing_target_attribute_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "missing target data arrays"):
            select_attribute_aware_baseline(
                strategy=ATTRIBUTE_FILTERED_RANDOM,
                attribute_metadata=self.metadata,
                metadata_indices=self.metadata_indices,
                class_labels=self.class_labels,
                generated_attribute_counts={"pose": {"running": 1000}},
                n_select=2,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
