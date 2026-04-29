# Pose Bug Compare Standard

- Standard failure threshold for future pose bug/failure-set comparisons: `acc_diff_threshold=0.02`.
- Failure rule: `slice_AP < overall_AP - acc_diff_threshold`.
- Fixed distribution threshold: `0.15`.
- Fixed rare-label threshold: `1.1`.
- Analysis tag: `manual`.
- Compared methods: `20` compact rows from `bugs_compare_pose.csv`; only the best `attribute` method per ratio is retained.

## Standard Result

- Main CSV: `bugs_compare_pose_acc_diff_0p02.csv`
- Detailed focused rows: `acc_diff_0p02/`
- Overall AP columns use the held-out test AP values from `report.md`, not the posthoc slice-summary precision values.
- The standard 0.02 CSV omits `acc_diff_threshold` because all rows use the same standard threshold.
- Base failure total before top-k: `12`
- Compared base failure cases per method: `12`
- Mean still failures over 20 rows: `7.05`
- Mean resolved failures over 20 rows: `4.95`

## Archived Threshold Sweep

The non-standard threshold results are preserved in one consolidated main-folder CSV:

- `bugs_compare_pose_threshold_sensitivity.csv`

This CSV has `120` rows: `20` compact method rows for each threshold `0.0`, `0.001`, `0.0025`, `0.005`, `0.01`, and `0.02`. Its Overall AP columns also use the held-out test AP values from `report.md`.

Detailed non-standard per-threshold folders remain archived under `archived_threshold_sweep/`:

- `acc_diff_0p0/`
- `acc_diff_0p001/`
- `acc_diff_0p0025/`
- `acc_diff_0p005/`
- `acc_diff_0p01/`
- `bugs_compare_pose_threshold_sensitivity.csv`
- `still_resolved_pivot.csv`

These are kept only for audit/comparison. Use `acc_diff_threshold=0.02` for future reported pose bug counts unless a later experiment explicitly changes the standard.
