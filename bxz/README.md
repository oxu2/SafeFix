# BLIP Pose Slice-Analysis Handoff

This folder is a handoff package for `bxz`.

The files here are copied from the current canonical outputs so the original paths used by the repo, scripts, and report remain unchanged.

## Read These First

1. `test_slice_stats.json`
   - Final clean held-out `test` slice analysis for the `bs=8` pose probe.
   - This is the main result file.
2. `test_metrics.json`
   - Overall held-out `test` metrics for the same pose probe.
   - Main number: `AP = 0.6370`.
3. `attribute_coco_pose_gemini_blip_val.json`
   - Clean BLIP eval-side attribute file for `val2017`.
   - Important: `weather=unknown` is gone in this file.
4. `attribute_coco_pose_gemini_train20k_blip.json`
   - Clean BLIP train-side attribute file for the first `20,000` valid `train2017` images.
5. `corpus_coco_pose_gemini.json`
   - Current attribute vocabulary definition used for the clean reruns.
6. `test_predictions.json`
   - Raw keypoint predictions used to compute `test_slice_stats.json`.
7. `qwen3_kitti_train5000_non_red_non_cloudy_matches.csv`
   - KITTI train-side Qwen3 rows that satisfy neither `color=red` nor
     `weather=cloudy`.
   - Columns mirror the earlier Qwen3 handoff CSV style:
     `row_idx,image_id,color,weather`.
   - Contains `4,508` data rows from the `5,000` KITTI train assignment rows.

## Main Findings

- This package corresponds to the `bs=8` BLIP pose probe:
  - original artifact root:
    `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/`
- The clean BLIP test slice analysis was regenerated after fixing the stale BLIP eval file.
- In the clean result, `weather=unknown` no longer appears.
- Main rare slices in `test_slice_stats.json`:
  - `pose=lying`
  - `clothes_color=brown`
  - `weather=rainy`
  - `weather=snowy`
  - `time_of_day=morning`

## Source Paths

- `corpus_coco_pose_gemini.json`
  - source: `save/corpus_coco_pose_gemini.json`
- `attribute_coco_pose_gemini_blip_val.json`
  - source: `save/attribute_coco_pose_gemini_blip_val.json`
  - generated from: `log_pose/blip_coco_pose_gemini_val_gpu2_20260412_171035.log`
- `attribute_coco_pose_gemini_train20k_blip.json`
  - source: `save/attribute_coco_pose_gemini_train20k_blip.json`
  - generated from: `log_pose/blip_coco_pose_gemini_train20k_gpu2_20260412_162538.log`
- `test_predictions.json`
  - source: `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/test_predictions.json`
- `test_metrics.json`
  - source: `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/test_metrics.json`
- `test_slice_stats.json`
  - source:
    `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/slice_analysis/blip_postcleanup_clean/test_slice_stats.json`
- `qwen3_kitti_train5000_non_red_non_cloudy_matches.csv`
  - source attribute CSV: `save/attribute_kitti_qwen3_8B.csv`
  - canonical copy: `save/qwen3_kitti_train5000_non_red_non_cloudy_matches.csv`
  - filter: keep rows where `color != red` and `weather != cloudy`
  - target counts in the source CSV: `492` rows satisfy `color=red OR
    weather=cloudy`; `12` rows satisfy both.

## Notes
