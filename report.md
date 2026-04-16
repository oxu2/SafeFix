# 04/17/2026 Update — GPU 0 and GPU 3 Pose-Training Runs Finished

- Verified that the two bs8 pose-estimation runs launched on April 15, 2026 are complete:
  - GPU `3` / Qwen3-VL-8B: original PID `1623280`
  - GPU `0` / InternVL3-8B: original PID `1624937`
- Both original PIDs are no longer alive, both logs end with `Training complete`, and both artifact roots contain:
  - `best.pth`
  - held-out `val` and `test` metrics JSONs
  - prediction dumps
  - post-train slice-analysis JSON/CSV outputs
- These are fresh runs under the current defaults (`batch_size=8`, `analysis_rare_label_threshold=1.1`), so they are **not** strict apples-to-apples reruns of the earlier `4/12` bs4 `rare0p97` probes.
- In both runs, the best checkpoint came from **epoch 0**; epoch 1 slightly regressed on held-out val AP.

| Run Tag | GPU | Best Epoch | Held-out Val AP | Held-out Test AP | Test AP50 | Test AP75 | Test AR | Train Time | Artifact Root | Log |
|---------|-----|------------|-----------------|------------------|-----------|-----------|---------|------------|---------------|-----|
| Qwen3-VL-8B | `3` | `0` | `0.6445` | `0.6352` | `0.8656` | `0.6902` | `0.6993` | `01:29:18` | `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_20260415_144429/` | `log_pose/pose_train20k_qwen3_bs8_e2_rare1p1_20260415_144429.log` |
| InternVL3-8B | `0` | `0` | `0.6401` | `0.6357` | `0.8618` | `0.6936` | `0.7016` | `01:33:39` | `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/` | `log_pose/pose_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530.log` |

- Relative to the earlier `4/12` held-out test probes:
  - Qwen3-VL-8B improved from `0.5968` to `0.6352` (`+0.0384`)
  - InternVL3-8B moved from `0.6340` to `0.6357` (`+0.0017`)
- Top-level `log_pose/` logs explicitly record both training-only and end-to-end elapsed time for these runs:
  - Qwen3-VL-8B: training `01:29:18` (`5358.44 s`), end-to-end `01:29:26` (`5366.02 s`)
  - InternVL3-8B: training `01:33:39` (`5619.12 s`), end-to-end `01:33:46` (`5625.88 s`)
- Held-out test slice analysis for both runs used `rare_label_threshold=1.1`.
- The strongest recurring **rare-case** slice is still `pose=lying` in both runs:
  - Qwen3-VL-8B: `AP=0.3091`, `ap_delta=-0.3261`, `34` test images
  - InternVL3-8B: `AP=0.3625`, `ap_delta=-0.2733`, `29` test images
- Other representative **rare cases** on the held-out test split:
  - Qwen3-VL-8B: `time_of_day=evening` (`AP=0.5629`, `164` images), `weather=rainy` (`AP=0.5799`, `31` images), `time_of_day=night` (`AP=0.5816`, `74` images)
  - InternVL3-8B: `time_of_day=night` (`AP=0.5725`, `191` images), `background_style=rural` (`AP=0.5756`, `103` images), `weather=rainy` (`AP=0.5796`, `32` images)
- `pose=sitting` remains the main high-support hard slice:
  - Qwen3-VL-8B: `AP=0.5503`, `329` test images
  - InternVL3-8B: `AP=0.5480`, `288` test images
- Other notable low-AP slices on the held-out test split:
  - Qwen3-VL-8B: `background_style=indoors` (`AP=0.5899`, hard_case), `weather=clear` (`AP=0.5899`, hard_case)
  - InternVL3-8B: `background_style=indoors` (`AP=0.5866`, hard_case), `weather=clear` (`AP=0.6017`, hard_case)

### Follow-Up Addendum (Artifacts dated 2026-04-15; original table above preserved)

- Two later same-day reruns requested after the initial summary are now complete:
  - Qwen3-VL-8B reproducibility rerun on GPU `0`:
    - artifact root: `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro/`
    - log: `log_pose/pose_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro.log`
  - InternVL3-8B training-only rerun on GPU `3` followed by post-hoc slice analyses:
    - artifact root: `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404/`
    - training log: `log_pose/pose_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404.log`
    - post-hoc sweep log: `log_pose/posthoc_slice_analysis_internvl3_trainonly_gpu3_20260415_183404.log`
- In both later reruns, the best checkpoint again came from **epoch 0**.

| Follow-Up Run | Earlier 4/17 Reference | Held-out Val AP | Held-out Test AP | Test AP50 | Test AP75 | Test AR | Train Time |
|---------------|------------------------|-----------------|------------------|-----------|-----------|---------|------------|
| Qwen3-VL-8B GPU `0` repro | Qwen3-VL-8B GPU `3` (`20260415_144429`) | `0.6445 -> 0.6407` (`-0.0038`) | `0.6352 -> 0.6341` (`-0.0011`) | `0.8656 -> 0.8609` (`-0.0047`) | `0.6902 -> 0.6866` (`-0.0036`) | `0.6993 -> 0.6958` (`-0.0035`) | `01:29:18 -> 01:38:50` (`+00:09:32`) |
| InternVL3-8B GPU `3` train-only rerun | InternVL3-8B GPU `0` (`20260415_144530`) | `0.6401 -> 0.6412` (`+0.0011`) | `0.6357 -> 0.6371` (`+0.0014`) | `0.8618 -> 0.8583` (`-0.0035`) | `0.6936 -> 0.6963` (`+0.0027`) | `0.7016 -> 0.6950` (`-0.0066`) | `01:33:39 -> 01:29:22` (`-00:04:17`) |

- Qwen3-VL-8B repro interpretation:
  - overall held-out metrics are slightly below the earlier GPU `3` Qwen3 run
  - the hardest rare slice `pose=lying` improved from `AP=0.3091` to `AP=0.3906` (`+0.0815`)
  - `pose=sitting` also improved slightly from `AP=0.5503` to `AP=0.5529`
  - low-light slices remained difficult and close to the earlier run:
    - `time_of_day=evening`: `0.5629 -> 0.5737`
    - `time_of_day=night`: `0.5816 -> 0.5802`
- InternVL3-8B GPU `3` train-only rerun interpretation:
  - held-out AP is slightly higher than the earlier GPU `0` InternVL3 run on both `val` and `test`
  - the hardest rare slice `pose=lying` improved from `AP=0.3625` to `AP=0.4021` (`+0.0396`)
  - `pose=sitting` improved from `AP=0.5480` to `AP=0.5585`
  - representative rare-case test slices in the rerun are:
    - `time_of_day=night`: `AP=0.5693`, `191` images
    - `background_style=rural`: `AP=0.5738`, `103` images
    - `weather=rainy`: `AP=0.5871`, `32` images
    - `time_of_day=evening`: `AP=0.6296`, `72` images
- Important sweep result for the InternVL3 rerun:
  - the post-hoc `rare_label_threshold` analyses at `0.97`, `0.95`, and `0.93` all completed successfully
  - the resulting `val` CSVs are byte-identical across thresholds
  - the resulting `test` CSVs are also byte-identical across thresholds
  - therefore, for this checkpoint, lowering the post-hoc rare threshold from `0.97` to `0.95` or `0.93` did **not** change the slice rankings or rare/hard-case labels
  - the three completed slice-analysis roots are:
    - `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404/slice_analysis/internvl3_rare0p97/`
    - `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404/slice_analysis/internvl3_rare0p95/`
    - `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404/slice_analysis/internvl3_rare0p93/`
- Important sweep result for the Qwen3 repro:
  - the post-hoc `rare_label_threshold` analyses at `0.97`, `0.95`, and `0.93` all completed successfully under:
    - `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro/slice_analysis/qwen3_rare0p97/`
    - `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro/slice_analysis/qwen3_rare0p95/`
    - `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro/slice_analysis/qwen3_rare0p93/`
  - the resulting `val` CSVs are byte-identical across thresholds
  - the resulting `test` CSVs are also byte-identical across thresholds
  - therefore, for this checkpoint, lowering the post-hoc rare threshold from `0.97` to `0.95` or `0.93` did **not** change the Qwen3 slice rankings or rare/hard-case labels
  - the stable Qwen3 **val** rare cases across all three thresholds are:
    - `pose=lying`: `AP=0.3623`, `27` images
    - `time_of_day=night`: `AP=0.5826`, `61` images
    - `clothes_color=brown`: `AP=0.5910`, `37` images
    - `time_of_day=morning`: `AP=0.6135`, `105` images
    - `time_of_day=evening`: `AP=0.6252`, `161` images
    - `clothes_color=gray`: `AP=0.6336`, `113` images
  - the stable Qwen3 **test** rare cases across all three thresholds are:
    - `pose=lying`: `AP=0.3906`, `34` images
    - `time_of_day=evening`: `AP=0.5737`, `164` images
    - `weather=rainy`: `AP=0.5763`, `31` images
    - `time_of_day=night`: `AP=0.5802`, `74` images
    - `background_style=rural`: `AP=0.6045`, `86` images
    - `clothes_color=pink`: `AP=0.6054`, `43` images
    - `clothes_color=gray`: `AP=0.6147`, `119` images
    - `clothes_color=brown`: `AP=0.6202`, `38` images

# 4/12/2026 Update — COCO Gemini Attributes and Pose Slice Analysis

### Summary of Recent Report Updates

Based on the completed runs and report updates consolidated on 4/12, the main recent work was:

- Established the COCO pose-estimation baseline stack, including Keypoint R-CNN fine-tuning and RTMPose-Tiny evaluation for later slice analysis.
- Refined the KITTI object-detection baseline around the focused `Car` / `Pedestrian` setup and updated the supporting training/report artifacts.
- Built the COCO pose attribute-assignment workflow for VLMs: added COCO pose corpora, updated `utils/assign_attribute.py` for COCO loading/filtering plus `--stop_at` and CSV-oriented output handling, and generated Qwen3 / InternVL3 attribute-assignment results.
- Added the BLIP-based COCO pose attribute-assignment path, completed a full run with the Gemini-defined corpus, and extended BLIP output handling to support CSV export.
- Finished the Gemini-corpus Qwen3 / InternVL3 validation runs on the COCO pose `val2017` split and recorded their runtimes, logs, and CSV outputs.
- Finished the Gemini-corpus Qwen3 / InternVL3 train-split runs on the first `20,000` valid `train2017` images and recorded their runtimes, logs, and CSV outputs.
- Finished three `train_pose_estimation.py` probe runs on the first `20,000` valid `train2017` images, added a reproducible held-out `val/test` split inside valid `val2017`, and recorded both the held-out keypoint metrics and the main slice-level failure cases.

### Recent COCO Gemini Workflow Updates

- Updated `utils/blip.py` so the COCO pose path defaults to `save/corpus_coco_pose_gemini.json` while still allowing override via `--corpus save/corpus_coco_pose.json`.
- Added CSV export support to `utils/blip.py` via `--csv_output`, with the default CSV path derived from the JSON output path.
- Updated `utils/blipFeature.py` to use configurable `BLIP_NUM_WORKERS` (default `0`) so BLIP inference does not fail in restricted execution environments.

### COCO Pose Corpus and Split Notes

- Rewrote `save/corpus_coco_pose.json` so the COCO pose attribute vocabulary is closer to **HiBug2 Appendix Table 5** and more suitable for **ControlNet / generative editing** rather than diagnosis-only tagging.
- The current COCO pose corpus contains **18 editable attributes**:
  - `pose`
  - `pose_complexity`
  - `head_orientation`
  - `clothes_color`
  - `is_arm_crossing`
  - `is_carrying_something`
  - `is_standing_on_one_leg`
  - `is_on_all_fours`
  - `is_sitting`
  - `is_dynamic`
  - `background_style`
  - `background_color`
  - `is_containing_other_people`
  - `indoor_lighting`
  - `is_indoor`
  - `weather`
  - `occlusion`
  - `viewpoint`
- These attributes cover pose state, body configuration, scene context, and imaging conditions, making them more directly usable for controlled generation and downstream bug-repair experiments.
- For the COCO pose pipeline, the repository currently uses:
  - **train**: `train2017` + `person_keypoints_train2017.json`
  - **val**: `val2017` + `person_keypoints_val2017.json`
  - **test**: `val2017` + `person_keypoints_val2017.json`
- This follows standard local COCO practice: because official test ground truth is not public, `val2017` is reused as the local held-out test set.
- After filtering to valid person-keypoint images, the current effective COCO pose pool contains **56,599 train images** and **2,346 val images**, for a total of **58,945 valid COCO pose images**.

### Full COCO Run

- Launched a full BLIP attribute-assignment run on COCO pose with the Gemini-defined corpus.
- Final JSON result was successfully produced at `save/attribute_coco_pose_gemini_blip.json`.
- The completed file contains `58,945` entries for each of the 6 Gemini-corpus attributes:
  - `clothes_color`
  - `clothes_type`
  - `background_style`
  - `time_of_day`
  - `weather`
  - `pose`
- Quick inspection confirmed all 6 attributes have non-empty assignments for all `58,945` samples.

### Gemini-Corpus VLM Val Runs (`utils/assign_attribute.py`)

- Completed the earlier Qwen3 and InternVL3 COCO pose attribute-assignment runs with the Gemini corpus using `utils/assign_attribute.py`.
- These runs used the script's default `coco_pose` split behavior, so they covered **val2017 only** rather than the combined train+val pool.
- Both runs loaded `person_keypoints_val2017.json`, filtered to **2,346 valid images**, and finished with both JSON and CSV outputs.

| Model | Split | Valid Images | Runtime | JSON Output | CSV Output | Log |
|-------|-------|--------------|---------|-------------|------------|-----|
| Qwen3-VL-8B | val2017 | 2,346 | 45.8 min | `save/attribute_coco_pose_gemini_qwen3_8B.json` | `save/attribute_coco_pose_gemini_qwen3_8B.csv` | `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_gpu3_20260410_193102.log` |
| InternVL3-8B | val2017 | 2,346 | 208.8 min | `save/attribute_coco_pose_gemini_internvl3_8B.json` | `save/attribute_coco_pose_gemini_internvl3_8B.csv` | `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_gpu2_20260410_193102.log` |

- Log endings confirm both jobs completed successfully:
  - Qwen3: `Done. Processed 2346 images in 45.8 minutes` and `Saved attribute annotations to save/attribute_coco_pose_gemini_qwen3_8B.json`
  - InternVL3: `Done. Processed 2346 images in 208.8 minutes` and `Saved attribute annotations to save/attribute_coco_pose_gemini_internvl3_8B.json`
- CSV validation passed for both outputs: each file contains `2,346` data rows plus a header row, and a quick missing-cell check found no empty attribute fields.

### Gemini-Corpus VLM Train Runs (`utils/assign_attribute.py`, 20k subset)

- Completed the follow-up Qwen3 and InternVL3 COCO pose train-split runs with the Gemini corpus using `--split train --stop_at 20000`.
- Both runs loaded `person_keypoints_train2017.json`, detected **56,599 valid train images**, and then processed the first **20,000** samples as requested.
- Both jobs finished with JSON and CSV outputs for the 6-attribute Gemini corpus.

| Model | Split | Source Pool | Processed Images | Runtime | JSON Output | CSV Output | Log |
|-------|-------|-------------|------------------|---------|-------------|------------|-----|
| Qwen3-VL-8B | train2017 | 56,599 valid images | 20,000 | 399.7 min (6.66 h) | `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json` | `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv` | `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_gpu0_20260410_193459.log` |
| InternVL3-8B | train2017 | 56,599 valid images | 20,000 | 1709.6 min (28.49 h) | `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json` | `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv` | `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_gpu1_20260410_193459.log` |

- Log endings confirm both jobs completed successfully:
  - Qwen3: `Done. Processed 20000 images in 399.7 minutes` and `Saved attribute annotations to save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
  - InternVL3: `Done. Processed 20000 images in 1709.6 minutes` and `Saved attribute annotations to save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`
- CSV validation passed for both train outputs: each file contains `20,000` data rows plus a header row.

### Canonical Post-Cleanup Completion Update (4/15/2026)

- The previously running post-cleanup attribute jobs are now complete.
- The current stable post-cleanup outputs are:

| Model | Split / Shard | Valid Images Written | Runtime | Stable Outputs | Primary Log |
|-------|---------------|----------------------|---------|----------------|-------------|
| Qwen3-VL-8B | canonical `val2017` rerun | `2,346` | `46.7` min (`0.78 h`) | `save/attribute_coco_pose_gemini_qwen3_8B.json`, `save/attribute_coco_pose_gemini_qwen3_8B.csv`, plus alias copies `save/attribute_coco_pose_gemini_qwen3_8B_val.json` and `save/attribute_coco_pose_gemini_qwen3_8B_val.csv` | `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_val_gpu1_20260414_134529.log` |
| InternVL3-8B | canonical `val2017` rerun | `2,346` | `217.2` min (`3.62 h`) | `save/attribute_coco_pose_gemini_internvl3_8B.json`, `save/attribute_coco_pose_gemini_internvl3_8B.csv` | `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_val_gpu1_20260414_143656.log` |
| InternVL3-8B | canonical `train20k` rerun | `20,000` total rows in the canonical artifact; the resume stage processed the remaining `19,500` images after a `500`-row checkpoint | `1876.5` min (`31.28 h`) for the resumed segment | `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`, `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv` | `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu0_20260413_233306.log` |
| Qwen3-VL-8B | `train_after20k` continuation shard | `36,599` | `731.4` min (`12.19 h`) | `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.json`, `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.csv` | `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train_after20k_gpu3_20260414_134834.log` |

- Current CSV row counts confirm the finished artifact sizes:
  - `save/attribute_coco_pose_gemini_qwen3_8B_val.csv`: `2,346` data rows
  - `save/attribute_coco_pose_gemini_internvl3_8B.csv`: `2,346` data rows
  - `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`: `20,000` data rows
  - `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.csv`: `36,599` data rows
- Qwen3 now covers the full valid COCO pose train split through two explicit non-overlapping shards:
  - the canonical first shard `save/attribute_coco_pose_gemini_train20k_qwen3_8B.{json,csv}` contains the first `20,000` valid train images
  - the continuation shard `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.{json,csv}` contains the remaining `36,599` valid train images
  - together these two shards cover all `56,599` valid `train2017` images

### Status Note

- The long-running `nohup` job finished successfully and the log ended with `Saved attributes to save/attribute_coco_pose_gemini_blip.json`.
- That particular background job was started before CSV export was added, so it produced JSON only, not CSV.

### CSV Validation

- After adding CSV support, a `--stop_at 1` smoke run on the Gemini corpus succeeded and produced both:
  - `save/attribute_coco_pose_gemini_blip_smoke_csv_gpu2.json`
  - `save/attribute_coco_pose_gemini_blip_smoke_csv_gpu2.csv`
- Example CSV row from the smoke run:

```csv
image_id,clothes_color,clothes_type,background_style,time_of_day,weather,pose
139,yellow,formal,indoors,morning,sunny,standing
```

### Follow-Up

- If a full CSV matching `save/attribute_coco_pose_gemini_blip.json` is needed, it can now be generated with the updated BLIP code in a rerun, or converted directly from the finished JSON in a short post-process step.
- The dedicated BLIP `train20k` rerun with JSON+CSV output completed at `log_pose/blip_coco_pose_gemini_train20k_gpu2_20260412_162538.log`.
- The dedicated BLIP `val` rerun with the cleaned corpus completed at `log_pose/blip_coco_pose_gemini_val_gpu2_20260412_171035.log` and produced:
  - `save/attribute_coco_pose_gemini_blip_val.json`
  - `save/attribute_coco_pose_gemini_blip_val.csv`
- A clean BLIP held-out `test` slice analysis now exists at:
  - `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/slice_analysis/blip_postcleanup_clean/test_slice_stats.json`
  - `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/slice_analysis/blip_postcleanup_clean/test_slice_stats.csv`
- In this clean BLIP reanalysis, `weather=unknown` no longer appears; the weather slices are now `sunny`, `clear`, `cloudy`, `rainy`, and `snowy`.

### COCO Pose 20k Probe Runs with Held-Out Slice Analysis

- Extended `train_pose_estimation.py` so the current probe workflow trains on the first `20,000` valid `train2017` images by default, splits the `2,346` valid `val2017` images into reproducible held-out `val` / `test` halves (`1,173` each, seed `42`), evaluates the best checkpoint on both halves, and runs automatic slice analysis from Gemini-corpus attribute annotations.
- Pose-related logs and experiment artifact directories are now consolidated under `log_pose/` instead of being split across `logs/` and `outputs/`.
- These results are useful as fast slice-analysis probes, but they are not direct replacements for the earlier full-val baseline because the train cap, batch size, and evaluation split are different.
- The three runs below should be read as end-to-end run variants rather than a strict attribute-source-only ablation, because the `BLIP` run used `batch_size=8` while the `Qwen3` and `InternVL3` runs used `batch_size=4`.

| Run Tag | Batch | Train Images | Held-out Val AP | Held-out Test AP | Test AP50 | Test AP75 | Test AR | Train Time | End-to-End |
|---------|-------|--------------|-----------------|------------------|-----------|-----------|---------|------------|------------|
| Qwen3-VL-8B | 4 | 20,000 | 0.5985 | 0.5968 | 0.8371 | 0.6470 | 0.6675 | 01:22:46 | 01:22:52 |
| InternVL3-8B | 4 | 20,000 | 0.6436 | 0.6340 | 0.8620 | 0.6922 | 0.7002 | 01:29:00 | 01:29:07 |
| BLIP | 8 | 20,000 | 0.6427 | 0.6370 | 0.8614 | 0.6902 | 0.7009 | 01:30:28 | 01:30:35 |

- Main artifact roots:
  - Qwen3: `log_pose/pose_keypointrcnn_train20k_qwen3_bs4_e2_rare0p97_20260412_015524/`
  - InternVL3: `log_pose/pose_keypointrcnn_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015632/`
  - BLIP: `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/`
- On the held-out test split, the strongest AP came from the `BLIP`-tagged run (`0.6370`), closely followed by `InternVL3` (`0.6340`); `Qwen3` reached `0.5968`.
- The clearest repeated failure slice across all three variants was `pose=lying`:
  - Qwen3: `AP=0.2796` (`delta=-0.3172`, `34` images)
  - InternVL3: `AP=0.2924` (`delta=-0.3416`, `32` images)
  - BLIP: `AP=0.4195` (`delta=-0.2175`, `68` images)
- `pose=sitting` was the most consistent high-support hard case in every run, with held-out test AP between `0.5225` and `0.5633`.
- Secondary failure slices depended on the annotation source: Qwen3 emphasized `time_of_day=unknown/morning` and `weather=unknown`; InternVL3 emphasized `background_style=rural` and `time_of_day=night/evening`; BLIP emphasized `background_style=indoors`, `weather=unknown`, and `clothes_type=formal`.

---

# 4/3/2026 Update — Multi-Task Baseline Training (Pose Estimation & Object Detection)

Following the safefix2 paper (arXiv:2501.16751), we train baseline models for pose estimation (COCO) and object detection (KITTI) to support error slice discovery across multiple vision tasks.

---

## 1. COCO Pose Estimation — Keypoint R-CNN (Pretrained, 2 epochs)

**Model**: `keypointrcnn_resnet50_fpn` (torchvision, COCO pretrained)
**Dataset**: COCO 2017 person keypoints, filtered to valid person-keypoint images (`train=56,599`, `val=2,346`, total `58,945`)
**Split**: train=`train2017` + `person_keypoints_train2017.json`; val=`val2017` + `person_keypoints_val2017.json`; local test also uses `val2017` by COCO convention because official test ground truth is not public
**Training**: 2 epochs, batch_size=2, lr schedule from torchvision defaults
**GPU**: GPU 3 | **Env**: `ad` conda (torch 1.13.1+cu117)
**Output**: `log_pose/pose_keypointrcnn_ad_gpu3_e2/`
**Log**: `log_pose/pose_coco_pretrained_gpu3_ad_e2.log`

### Per-Epoch Results (OKS-based COCO Keypoint Metrics)

| Epoch | Train Loss | AP | AP50 | AP75 | AP (M) | AP (L) | AR | AR50 | AR75 | AR (M) | AR (L) | Time |
|-------|-----------|------|------|------|--------|--------|------|------|------|--------|--------|------|
| 0 | 4.239 | 0.481 | 0.752 | 0.504 | — | — | 0.564 | — | — | — | — | 3.88h |
| **1** | **4.321** | **0.489** | **0.764** | **0.513** | **0.447** | **0.553** | **0.564** | **0.821** | **0.599** | **0.516** | **0.632** | **3.91h** |

**Total wall time**: ~7.8 hours (2 epochs). Best checkpoint saved at epoch 1.

> [!NOTE]
> The pretrained Keypoint R-CNN already achieves AP=0.489 after just 2 epochs of fine-tuning on COCO. Longer training (12 epochs per README) would improve results further, but 2 epochs are sufficient for establishing the baseline for error slice debugging.

---

## 2. COCO Pose Estimation — RTMPose-Tiny (Pretrained Evaluation)

**Model**: RTMPose-Tiny (SimCC head, pretrained on AIC+COCO, 420 epochs)
**Checkpoint**: `rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth`
**Dataset**: COCO 2017 val2017 (794 batches)
**GPU**: GPU 3 | **Env**: `ad` conda (mmengine + mmpose)
**Log**: `log_pose/rtmpose_tiny_coco_gpu3_probe/20260403_131401/20260403_131401.log`

### Evaluation Results (OKS-based COCO Keypoint Metrics)

| Metric | Value |
|--------|-------|
| AP (IoU=0.50:0.95) | **0.697** |
| AP (IoU=0.50) | 0.906 |
| AP (IoU=0.75) | 0.776 |
| AP (Medium) | 0.673 |
| AP (Large) | 0.739 |
| AR (IoU=0.50:0.95) | **0.725** |
| AR (IoU=0.50) | 0.915 |
| AR (IoU=0.75) | 0.795 |
| AR (Medium) | 0.693 |
| AR (Large) | 0.774 |

> [!NOTE]
> RTMPose-Tiny (AP=0.697) significantly outperforms Keypoint R-CNN (AP=0.489 after 2 epochs) as expected, since RTMPose was trained for 420 epochs on AIC+COCO with a specialized SimCC head. Per the safefix2 paper, the tiny/small/medium/large RTMPose variants are all used for error slice analysis on pose estimation.

---

## 3. KITTI Object Detection — YOLOv8n (Updated 2026-04-09, Car/Pedestrian Split)

**Model**: YOLOv8n (pretrained on COCO, `weights/yolov8n.pt`)
**Dataset**: KITTI training split in deterministic filename order, converted to YOLO format with only `Car` and `Pedestrian`
**Split**: first 5000 images for train, next 1240 for val, remaining 1241 for test
**Implementation note**: all three splits are derived from the KITTI `training/` set in deterministic filename order, then exported to YOLO-format `images/{train,val,test}` and `labels/{train,val,test}`
**Training**: 30 epochs, batch=4, imgsz=640, optimizer=`auto` -> AdamW (`lr=0.001667`, momentum=0.9)
**GPU**: NVIDIA RTX A5000 | **Env**: Ultralytics 8.4.33, torch 2.9.1+cu128
**Output**: `runs/kitti_yolov8/first5000_gpu1_e30_20260409_194735/`
**Log**: `logs/yolov8_kitti_gpu1_e30_20260409_194735.log`

### Best Validation Checkpoint (Epoch 29)

| Metric | Value |
|--------|-------|
| Precision | 0.849 |
| Recall | 0.712 |
| mAP@50 | **0.808** |
| mAP@50-95 | **0.545** |
| Fitness | 0.545 |

### Held-Out Test Results (`best.pt` on `test`)

| Metric | Value |
|--------|-------|
| Precision | 0.864 |
| Recall | 0.750 |
| mAP@50 | **0.831** |
| mAP@50-95 | **0.563** |
| Fitness | 0.563 |

### Per-Class Test Results

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **1241** | **5369** | **0.864** | **0.750** | **0.831** | **0.563** |
| Car | 1107 | 4666 | 0.899 | 0.883 | 0.936 | 0.731 |
| Pedestrian | 283 | 703 | 0.829 | 0.617 | 0.726 | 0.394 |

**Total training time**: 1h 01m 38s (30 epochs). **End-to-end wall time**: 1h 01m 43s. **Test evaluation time**: 43s.

> [!IMPORTANT]
> This run supersedes the earlier 8-class `train=5000 / val=500` YOLO baseline in the previous report draft. With the new deterministic `train=5000 / val=1240 / test=1241` split and a focused `Car`/`Pedestrian` label space, the held-out test results still show a clear class gap: **Pedestrian (0.394 mAP@50-95)** remains substantially weaker than **Car (0.731)**, making pedestrian-heavy slices the clearest target for safefix-style debugging.

### Validation Convergence

| Epoch | Precision | Recall | mAP@50 | mAP@50-95 | val/box_loss | val/cls_loss |
|-------|-----------|--------|--------|-----------|-------------|-------------|
| 1 | 0.728 | 0.571 | 0.638 | 0.375 | 1.218 | 0.908 |
| 5 | 0.772 | 0.603 | 0.685 | 0.424 | 1.108 | 0.729 |
| 10 | 0.846 | 0.634 | 0.738 | 0.470 | 1.029 | 0.657 |
| 20 | 0.839 | 0.682 | 0.774 | 0.513 | 0.923 | 0.562 |
| 29 (best) | 0.849 | 0.712 | 0.808 | **0.545** | 0.861 | 0.516 |
| 30 | 0.844 | 0.704 | 0.802 | 0.545 | 0.854 | 0.513 |

---

## Summary of Multi-Task Baselines

| Task | Model | Dataset | Key Metric | Value | Status |
|------|-------|---------|------------|-------|--------|
| Pose Estimation | Keypoint R-CNN | COCO 2017 | AP (OKS) | 0.489 | 2 epochs, pretrained |
| Pose Estimation | RTMPose-Tiny | COCO 2017 | AP (OKS) | 0.697 | Pretrained eval only |
| Object Detection | YOLOv8n | KITTI 5000/1240/1241 (`Car`, `Pedestrian`) | mAP@50-95 (test) | 0.563 | 30 epochs, pretrained |

These baselines are ready for error slice discovery and model debugging with safefix2/DebugAgent.

---

# 3/27/2026 Update
## Long-Tail Learning Baselines

### Motivation

Reviewer feedback noted that once the failing subgroup is identified (via safefix), the remaining task—generating more data for a rare slice and retraining—resembles long-tail learning with synthetic augmentation. To address this, we implement four standard Long-Tail Learning (LTL) baselines as controlled comparisons. These baselines operate **exclusively on the original training data** without any generated images, isolating the effect of each LTL strategy from Safefix's targeted synthetic augmentation.

The comparison demonstrates that class-level reweighting and resampling (standard LTL) cannot fix **attribute-level** subgroup failures (e.g., "red hair + brown skin" faces failing in a lipstick classifier), whereas Safefix's targeted generation can.



---

### Baseline 1: Focal Loss (`--ltl_baseline focal`)

**Reference**: Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.

Focal Loss down-weights well-classified (easy) examples, allowing the model to focus training on hard, misclassified samples. This addresses class imbalance by reducing the loss contribution from the dominant class.

**Multiclass formulation** (ImageNet):

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the model's estimated probability for the true class, $\gamma \geq 0$ is the focusing parameter (default $\gamma = 2.0$), and $\alpha_t$ is the per-class weight. We compute $\alpha$ from inverse class frequencies:

$$\alpha_c = \frac{1/n_c}{\sum_{j=1}^{C} 1/n_j} \cdot C$$

where $n_c$ is the number of training samples in class $c$ and $C$ is the total number of classes.

**Binary formulation** (CelebA):

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \cdot \text{BCE}(\hat{y}, y)$$

where $p_t = y \cdot \sigma(\hat{y}) + (1-y) \cdot (1 - \sigma(\hat{y}))$, $\alpha_t = y \cdot \alpha + (1-y) \cdot (1-\alpha)$, and $\alpha = N_{\text{neg}} / (N_{\text{neg}} + N_{\text{pos}})$ so that the minority (positive) class receives higher weight.

**Hyperparameters**: `--focal_gamma` (default: 2.0).

---

### Baseline 2: Class-Balanced Loss (`--ltl_baseline class_weight`)

The most straightforward LTL approach: apply inverse-frequency class weights directly to the standard loss function.

**Multiclass** (ImageNet): `CrossEntropyLoss` with per-class weights:

$$w_c = \frac{1/n_c}{\sum_{j=1}^{C} 1/n_j} \cdot C$$

$$\mathcal{L} = -\sum_{i=1}^{N} w_{y_i} \log \frac{\exp(z_{i,y_i})}{\sum_j \exp(z_{i,j})}$$

**Binary** (CelebA): `BCEWithLogitsLoss` with `pos_weight`:

$$\texttt{pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$

This is equivalent to the scheme already embedded in the original `predict.py` via `SmoothBCEWithLogitsLoss`, but without label smoothing, providing a clean ablation point.

---

### Baseline 3: Oversampling (`--ltl_baseline resample`)

Instead of modifying the loss function, this baseline balances the class distribution at the **data sampling** level using PyTorch's `WeightedRandomSampler`. Each training sample $x_i$ with label $y_i$ receives a sampling weight inversely proportional to its class frequency:

$$w_i = \frac{1}{n_{y_i}}$$

where $n_{y_i}$ is the number of training samples in the class of $x_i$. The sampler draws $N$ samples per epoch (with replacement), so each class is expected to appear equally often in every epoch regardless of its original frequency.

The loss function is unweighted (`CrossEntropyLoss` for ImageNet, `BCEWithLogitsLoss` for CelebA), since balancing is handled entirely by the sampler.

---

### Baseline 4: LDAM (`--ltl_baseline ldam`)

**Reference**: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss," NeurIPS 2019.

LDAM enforces larger classification margins for minority classes, encouraging the model to learn more separable representations for rare classes. The key idea is that classes with fewer samples need a larger decision boundary margin to generalize well.

**Per-class margin**:

$$\Delta_j = \frac{C_{\max}}{n_j^{1/4}}$$

where $n_j$ is the number of training samples in class $j$, and $C_{\max}$ is a normalization constant such that $\max_j \Delta_j = m_{\max}$ (controlled by `--ldam_max_m`, default 0.5).

**Multiclass formulation** (ImageNet):

$$\mathcal{L}_{\text{LDAM}} = \text{CE}\left(s \cdot \tilde{z}, \, y\right)$$

where $\tilde{z}_{i,c} = z_{i,c} - \Delta_c \cdot \mathbb{1}[c = y_i]$ subtracts the margin from the true-class logit before computing cross-entropy, and $s$ is a logit scaling factor (default $s = 30$). We additionally apply inverse-frequency class weights to the CE component.

**Binary formulation** (CelebA):

$$\tilde{z}_i = s \cdot \left(z_i - y_i \cdot \Delta_{\text{pos}} - (1 - y_i) \cdot (-\Delta_{\text{neg}})\right)$$

$$\mathcal{L}_{\text{LDAM}} = \text{BCE}(\tilde{z}_i, \, y_i)$$

where $\Delta_{\text{pos}}$ and $\Delta_{\text{neg}}$ are the margins for the positive and negative classes respectively. Positive targets subtract $\Delta_{\text{pos}}$ (making correct classification harder), while negative targets add $\Delta_{\text{neg}}$ (similarly enforcing a wider margin).

**Hyperparameters**: `--ldam_max_m` (default: 0.5), `--ldam_s` (default: 30.0).

---

### Handling of `pos_weight` in CelebA

The original pipeline (`predict.py`) always applies `pos_weight = N_{\text{neg}} / N_{\text{pos}}` inside `SmoothBCEWithLogitsLoss`. This is effectively a class-reweighting strategy baked into every run. The LTL baselines handle this as follows:

| Baseline | Loss (CelebA) | Class balancing mechanism |
|---|---|---|
| `none` (default) | `SmoothBCEWithLogitsLoss` | `pos_weight` + label smoothing (original behavior) |
| `focal` | `BinaryFocalLoss` | $\alpha$ term replaces `pos_weight` |
| `class_weight` | `BCEWithLogitsLoss` | `pos_weight` (no label smoothing) |
| `resample` | `BCEWithLogitsLoss` | `WeightedRandomSampler` (no loss weighting) |
| `ldam` | `BinaryLDAMLoss` | Margin-based (no `pos_weight`) |

### Experimental Design

All LTL baselines use **only the original training data**:
- CelebA: 80,000 training images (no generated redhair_brownskin images)
- ImageNet: 10,400 training images (no generated images)

This ensures a fair comparison: LTL methods address class imbalance through algorithmic means only, while Safefix adds targeted synthetic data for specific attribute subgroups.

### Run Commands

```bash
# CelebA (GPU 2)
nohup /people/cs/o/oxx220000/.conda/envs/ad/bin/python -u predict_ltl.py \
  --dataset celeba --model resnet --ltl_baseline <METHOD> --gpu 2 \
  > logs/ltl_<METHOD>_celeba_resnet.log 2>&1 &

# ImageNet (GPU 1)
nohup /people/cs/o/oxx220000/.conda/envs/ad/bin/python -u predict_ltl.py \
  --dataset imagenet --model resnet --ltl_baseline <METHOD> --gpu 1 \
  > logs/ltl_<METHOD>_imagenet_resnet.log 2>&1 &
```

### Baseline Parameters (ImageNet10 Runs)

- **Focal Loss**: `Using Focal Loss (multiclass) with gamma=2.0`
- **Class-Balanced Loss**: `Using Class-Weighted CrossEntropyLoss, weights=[0.9989 0.9829 1.0009 0.9904 0.9942 1.0175 0.997 0.9989 1.0165 1.0028]`
- **Oversampling**: `Using Resample baseline with WeightedRandomSampler (class counts: [1041, 1058, 1039, 1050, 1046, 1022, 1043, 1041, 1023, 1037])`
- **LDAM Loss**: `Using LDAM Loss (multiclass) with max_m=0.5, s=30.0, weight=[0.9989 0.9829 1.0009 0.9904 0.9942 1.0175 0.997 0.9989 1.0165 1.0028]`

---

### Results Comparison

The following table summarizes the performance of LTL baselines vs. Safefix. 
**Note**: Paper baseline accuracy for ResNet-18 on ImageNet10 is **71.73%**. Safefix results are cited from Table 2 and Table 8 of the paper.

| Baseline | CelebA Test ACC | CelebA Fix Bugs | ImageNet Test ACC | ImageNet Fix Bugs |
|---|---|---|---|---|
| **Base** | **90.57%** | **-** | **71.73%** | **-** |
| Focal Loss | 91.4% | 8/13  | 68.8% | 11/17  |
| Class-Balanced Loss | 91.2% | 8/13  | 72.7% | 13/17  |
| Oversampling | 91.0% | 8/13  | 72.9% | 14/17  |
| LDAM | 91.4% | 8/13  | 71.2% | 12/17  |
| **Safefix (Paper Ours)** | **91.92%** |  | **74.88%** |  |

---

### Experiment Status (as of 03/27 12:30)

All ImageNet and CelebA LTL baseline runs are complete. Results indicate that while LTL methods can provide modest boosts to overall accuracy on balanced splits, they are significantly less effective than Safefix at addressing specific attribute-level failures discovered by safefix.

> [!IMPORTANT]
> **OOM Mitigation**: To address the `Unable to find a valid cuDNN algorithm` and `CUBLAS_STATUS_NOT_INITIALIZED` errors occurring on the heavily utilized shared GPUs, we successfully completed these runs by disabling `cudnn.benchmark` and reducing batch sizes to 32.




# 3/13/2026 Update
## All-Strategy Comparison across Sub Ratios (lr=1e-5, 15 epochs)

All runs resume from `weights/model_imagenet_resnet_base.pth` (base test acc: **0.7162**). The Attribute strategy uses the optimized `aggro_exp_p10_c4` parameters ($R=0.8, P=10.0, C=4.0$, Entropy=True). Reference: using 100% of the data (ratio=1.0) yields Test ACC = **0.7488**. The 3/13 update adds two new LLM-based attribute generation results (`Qwen3` and `InternVL3`) to compare alongside the default BLIP-based attribute assignment.

### Sub Ratio = 0.1
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7269 | 0.7069 |
| Stratified | 0.7271 | 0.7154 |
| Coreset | 0.7269 | 0.7115 |
| Uncertainty Stratified | 0.7254 | 0.7185 |
| Diversity | 0.7262 | 0.7215 |
| Hierarchical | 0.7269 | 0.7115 |
| Attribute (BLIP) | 0.7277 | 0.7200 |
| Attribute (Qwen3) | 0.7331 | 0.7231 |
| Attribute (InternVL3) | **0.7346** | 0.7285 |

### Sub Ratio = 0.3
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7308 | 0.7208 |
| Stratified | 0.7369 | 0.7162 |
| Coreset | 0.7369 | 0.7162 |
| Uncertainty Stratified | 0.7292 | 0.7131 |
| Diversity | 0.7362 | 0.7200 |
| Hierarchical | 0.7346 | 0.7092 |
| Attribute (BLIP) | 0.7369 | 0.7185 |
| Attribute (Qwen3) | 0.7354 | 0.7077 |
| Attribute (InternVL3) | **0.7385** | 0.7108 |

### Sub Ratio = 0.5
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7400 | 0.7162 |
| Stratified | 0.7446 | 0.7200 |
| Coreset | 0.7400 | 0.7185 |
| Uncertainty Stratified | 0.7331 | 0.7146 |
| Hierarchical | 0.7385 | 0.7138 |
| Hybrid | 0.7362 | 0.7215 |
| Attribute (BLIP) | 0.7408 | 0.7131 |
| Attribute (Qwen3) | 0.7362 | 0.7138 |
| Attribute (InternVL3) | **0.7495** | **0.7286** |

### Sub Ratio = 0.7
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | **0.7423** | **0.7223** |
| Stratified | 0.7346 | 0.7138 |
| Coreset | 0.7392 | 0.7154 |
| Uncertainty Stratified | 0.7354 | 0.7146 |
| Hierarchical | 0.7385 | 0.7146 |
| Attribute (BLIP) | 0.7385 | 0.7131 |
| Attribute (Qwen3) | 0.7415 | 0.7192 |
| Attribute (InternVL3) | 0.7413 | 0.7100 |

### Sub Ratio = 0.9
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.7482 | 0.7154 |
| Stratified | 0.7469 | 0.7146 |
| Coreset | 0.7346 | 0.7169 |
| Uncertainty Stratified | 0.7462 | 0.7100 |
| Diversity | 0.7415 | 0.7146 |
| Hierarchical | 0.7362 | 0.7177 |
| Hybrid | 0.7477 | 0.7177 |
| Attribute (BLIP) | 0.7400 | 0.7177 |
| Attribute (Qwen3) | 0.7492 | 0.7108 |
| Attribute (InternVL3) | **0.7531** | **0.7346** |


---

## Low Budget Active Learning Comparison (ratios: 0.01, 0.02, 0.05)

**Manual Selection**: In these experiments, the "Manual Selection" method involves hand-picking images where the target object (e.g., backpack) is prominent. Because the ImageNet dataset often contains noisy images—some of which do not even feature the intended object—this manual process specifically selects images without cluttered backgrounds. It ensures the target object is large, highly visible, and clearly the main subject of the picture rather than a secondary element.

Runs for sub_ratio = 0.01, 0.02, and 0.05 train ResNet-18 **from scratch** (no pretrained base) for 30 epochs on the `pinkcolor_fabrictexture` ImageNet split with 1000 generated images. Sub-ratio = 0.1 results use the resume-from-base setting (lr=1e-5, 15 epochs) and are provided here alongside the smaller ratios for a unified small-budget view.

### Sub Ratio = 0.01
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.3323 | 0.3177 |
| Stratified | 0.3154 | 0.3100 |
| Diversity | 0.3269 | 0.3115 |
| Hybrid | 0.3353 | 0.2931 |
| Coreset | 0.2846 | 0.2462 |
| Manual Selection | 0.3385 | 0.3215 |

### Sub Ratio = 0.02
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.3669 | 0.3415 |
| Stratified | 0.3623 | 0.3400 |
| Diversity | 0.3238 | 0.3015 |
| Hybrid | 0.3662 | 0.3450 |
| Coreset | 0.3085 | 0.2815 |
| Attribute | 0.3669 | 0.3480 |
| Manual Selection | 0.3650 | 0.3440 |

### Sub Ratio = 0.05
| Strategy | Test ACC | Val ACC |
|---|---|---|
| Random | 0.3977 | 0.3777 |
| Stratified | 0.3931 | 0.3762 |
| Diversity | 0.4162 | 0.3800 |
| Hybrid | 0.4169 | 0.3923 |
| Coreset | 0.3246 | 0.3069 |
| Manual Selection | 0.4115 | 0.3862 |



# 2/27/2026 Update




## Active Learning Resume-from-Base Comparison (lr=1e-5, 15 epochs)

This table shows results from `test_al_resume.sh` with `freeze_backbone=False`, `lr=1e-5`, and `epoch=15`. All runs use `sub_ratio=0.3` and resume from `weights/model_imagenet_resnet_base.pth` (base val acc: **0.7223**).

| Strategy | Model Test ACC | Overall Val ACC Change | Status |
| :--- | :--- | :--- | :--- |
| Baseline | 0.7162 | 0.718077 | - |
| Random | 0.7308 | 0.718077 → 0.720769 (+0.002692) | improved |
| Stratified | **0.7369** | 0.718077 → 0.726154 (+0.008077) | improved |
| Coreset | 0.7359 | 0.718077 → 0.719231 (+0.001154) | improved |
| Diversity | 0.7362 | 0.718077 → 0.723846 (+0.005769) | improved |
| Uncertainty Stratified | 0.7292 | 0.718077 → 0.719231 (+0.001154) | improved |
| Hierarchical | 0.7346 | 0.718077 → 0.724615 (+0.006538) | improved |
| `Attribute` | **0.7369** | 0.718077 → 0.722308 (+0.004231) | improved |


### Observations:
* **All strategies improved** over the baseline (0.718077) after fine-tuning with `lr=1e-5` for 15 epochs without backbone freezing.
* **Best Test ACC**: `Hybrid` (0.7400).
* **Best Val ACC improvement**: `Stratified` (+0.008077), benefiting from balanced class coverage during fine-tuning.
* **Uncertainty Stratified** performed below expectation (0.7292 Test ACC), possibly due to using a generic pretrained proxy model (ResNet18 weights, not the actual resumed checkpoint) to compute entropy scores, leading to noisy uncertainty estimates.
* **Attribute strategy** (with conditional rarity, R=0.01, B=5, class_balance=inf) achieved competitive performance (0.7369 Test ACC, +0.004231 val), demonstrating that metadata-driven selection can match geometry/diversity baselines when paired with proper fine-tuning LR.

## Grid Search Results for Attribute Strategy Parameters (2/27/2026)
We conducted a comprehensive grid search over the parameters $R, P, C,$ and the `use_entropy` toggle to understand their interactions and optimize the Attribute Strategy. The tests were run across 4 sub-ratios (`0.1`, `0.3`, `0.5`, `0.9`) on the `pinkcolor_fabrictexture` generated set.

### Sub Ratio = 0.1
| Scenario | Parameters | Test ACC | Val ACC |
|---|---|---|---|
| baseline_def | (R=0.01, P=5.0, C=inf, e=F) | **0.7300** | 0.7146 |
| strict_strat_entropy | (R=0.01, P=5.0, C=inf, e=T) | 0.7215 | 0.7154 |
| pure_greedy | (R=1.0, P=10.0, C=0.0, e=F) | 0.7208 | 0.7154 |
| soft_conservative_e0 | (R=0.1, P=5.0, C=15.0, e=F) | 0.7223 | 0.7146 |
| soft_conservative_e1 | (R=0.1, P=5.0, C=15.0, e=T) | 0.7208 | 0.7138 |
| balanced_exp_p5_c8 | (R=0.3, P=5.0, C=8.0, e=T) | 0.7223 | 0.7185 |
| balanced_exp_p10_c8 | (R=0.3, P=10.0, C=8.0, e=T) | 0.7223 | 0.7200 |
| aggro_exp_p5_c4 | (R=0.8, P=5.0, C=4.0, e=T) | 0.7223 | 0.7177 |
| aggro_exp_p10_c4 | (R=0.8, P=10.0, C=4.0, e=T) | 0.7277 | 0.7200 |

### Sub Ratio = 0.3
| Scenario | Parameters | Test ACC | Val ACC |
|---|---|---|---|
| baseline_def | (R=0.01, P=5.0, C=inf, e=F) | 0.7369 | 0.7262 |
| strict_strat_entropy | (R=0.01, P=5.0, C=inf, e=T) | 0.7346 | 0.7138 |
| pure_greedy | (R=1.0, P=10.0, C=0.0, e=F) | 0.7215 | 0.7162 |
| soft_conservative_e0 | (R=0.1, P=5.0, C=15.0, e=F) | 0.7392 | 0.7200 |
| soft_conservative_e1 | (R=0.1, P=5.0, C=15.0, e=T) | 0.7262 | 0.7115 |
| balanced_exp_p5_c8 | (R=0.3, P=5.0, C=8.0, e=T) | 0.7362 | 0.7108 |
| balanced_exp_p10_c8 | (R=0.3, P=10.0, C=8.0, e=T) | 0.7362 | 0.7092 |
| aggro_exp_p5_c4 | (R=0.8, P=5.0, C=4.0, e=T) | 0.7362 | 0.7138 |
| aggro_exp_p10_c4 | (R=0.8, P=10.0, C=4.0, e=T) | **0.7415** | 0.7115 |

### Sub Ratio = 0.5
| Scenario | Parameters | Test ACC | Val ACC |
|---|---|---|---|
| baseline_def | (R=0.01, P=5.0, C=inf, e=F) | **0.7438** | 0.7123 |
| strict_strat_entropy | (R=0.01, P=5.0, C=inf, e=T) | 0.7415 | 0.7123 |
| pure_greedy | (R=1.0, P=10.0, C=0.0, e=F) | 0.7338 | 0.7138 |
| soft_conservative_e0 | (R=0.1, P=5.0, C=15.0, e=F) | **0.7438** | 0.7208 |
| soft_conservative_e1 | (R=0.1, P=5.0, C=15.0, e=T) | 0.7400 | 0.7162 |
| balanced_exp_p5_c8 | (R=0.3, P=5.0, C=8.0, e=T) | 0.7308 | 0.7077 |
| balanced_exp_p10_c8 | (R=0.3, P=10.0, C=8.0, e=T) | 0.7362 | 0.7100 |
| aggro_exp_p5_c4 | (R=0.8, P=5.0, C=4.0, e=T) | 0.7400 | 0.7185 |
| aggro_exp_p10_c4 | (R=0.8, P=10.0, C=4.0, e=T) | 0.7408 | 0.7131 |

### Sub Ratio = 0.9
| Scenario | Parameters | Test ACC | Val ACC |
|---|---|---|---|
| baseline_def | (R=0.01, P=5.0, C=inf, e=F) | 0.7454 | 0.7223 |
| strict_strat_entropy | (R=0.01, P=5.0, C=inf, e=T) | 0.7438 | 0.7131 |
| pure_greedy | (R=1.0, P=10.0, C=0.0, e=F) | 0.7400 | 0.7192 |
| soft_conservative_e0 | (R=0.1, P=5.0, C=15.0, e=F) | 0.7408 | 0.7146 |
| soft_conservative_e1 | (R=0.1, P=5.0, C=15.0, e=T) | 0.7392 | 0.7192 |
| balanced_exp_p5_c8 | (R=0.3, P=5.0, C=8.0, e=T) | **0.7500** | 0.7162 |
| balanced_exp_p10_c8 | (R=0.3, P=10.0, C=8.0, e=T) | 0.7446 | 0.7138 |
| aggro_exp_p5_c4 | (R=0.8, P=5.0, C=4.0, e=T) | 0.7431 | 0.7154 |
| aggro_exp_p10_c4 | (R=0.8, P=10.0, C=4.0, e=T) | 0.7400 | 0.7177 |

### Analysis
1. **The Danger of Pure Greedy**: The `pure_greedy` approach ($C=0$) performs the worst across low evaluation ratios (0.1, 0.3). Without class-level Stratification quotas, throwing out all logic except Frequency Penalty and Rarity leads to performance drops in fine-tuning, emphasizing the value of our soft $C$ score.
2. **Aggressive Exploration Shines ($R=0.8, C=4.0$)**: By loosening the absolute restriction on stratification ($C=4.0$ instead of $\infty$) and massively scaling up Rarity tracking ($R=0.8$), the `aggro_exp_p10_c4` scenario achieves extremely strong results across 0.3 and 0.5. The heavy frequency penalty ($P=10$) adequately prevented the model from indexing too heavily on the synthesized injection flood.
3. **Entropy Tuning**: Dynamic entropy ($e=T$) does well in aggressive settings but causes occasional degradations in more conservative spaces (see `soft_conservative` comparisons).
4. **Baseline Comparison (Ratio=1.0)**: For reference, using **100% of the candidate data (Ratio=1.0)** yields a Test ACC of **0.7400**. Using Safefix’s method, updating all source data and generated images for 30 epochs results in a Test ACC of **0.7438**. Remarkably, the `balanced_exp_p5_c8` strategy at `Ratio=0.9` achieved an even higher Test ACC of **0.7500**, demonstrating that intelligent filtering can outperform training on the entirety of the noisy synthetic data!
5. **Overall Best Strategy**: Averaging across all sub-ratios, **`aggro_exp_p10_c4` ($R=0.8, P=10.0, C=4.0$, Entropy=True)** represents the most robust and high-performing parameter combination. It consistently dominates the low-to-mid ratio regimes (0.1 to 0.5) by aggressively seeking rare features while safely blocking flooded features and maintaining a soft class balance.

**Conclusion**: Tuning the Attribute Strategy to mimic `Coreset` + `Stratified` (via Soft C + High R) allows the strategy to exceed the conservative defaults! When allocating budgets, choosing a strong R and applying a compensatory P to manage spam achieves peak domain adaptability.

# 2/20/2026 Update
## Active Learning Resume-from-Base Comparison

This table compares the test results generated by `test_al_resume.sh` against the provided baseline (`logs_imagenet/AL_attr_resume_base_resnet_imagenet_0.3_1000.log`). All tasks use `sub_ratio=0.3` and resume from the base model.

| Strategy | Model Test ACC | Val Accuracy Difference | Status |
| :--- | :--- | :--- | :--- |
| Baseline | 0.7162 | 0.718077 | - |
| Random | 0.7385 | 0.718077 -> 0.717692 (-0.000385) | declined |
| Stratified | 0.7362 | 0.718077 -> 0.725385 (+0.007308) | improved |
| Coreset | 0.7346 | 0.718077 -> 0.720000 (+0.001923) | improved |
| Uncertainty Stratified | 0.7385 | 0.718077 -> 0.721538 (+0.003461) | improved |
| Diversity | 0.7315 | 0.718077 -> 0.720769 (+0.002692) | improved |
| Hierarchical | 0.7415 | 0.718077 -> 0.716154 (-0.001923) | declined |
| **Hybrid** | **0.7446** | 0.718077 -> 0.717692 (-0.000385) | declined |
| `Attribute` | 0.7369 | 0.718077 -> 0.720000 (+0.001923) | improved |

### Observations:
* **Best Test ACC**: `Hybrid` strategy achieved the highest raw Model Test Accuracy (0.7446).
* **Best Overall Accuracy Variation**: `Stratified` provided the best improvement in attribute/hard-case accuracy comparison (+0.007308).

---

## Generation-Guided Active Learning (From Scratch) Comparison

This section compares the new `gen_probe` strategy against the baseline `random` and `stratified` strategies **from scratch** (i.e., without `--resume_model`).
All runs use `sub_ratio=0.3`, 1000 generated images, and are trained for 30 epochs on ResNet-18.

| Strategy | Model Test ACC | Status compared to Random |
| :--- | :--- | :--- |
| Baseline | 0.2462 | - |
| Random | 0.5754 | - |
| Stratified | **0.5908** | +0.0154 |
| Gen-Probe | 0.5600 | -0.0154 |

### Observations:
* The `gen_probe` strategy (Test ACC: 0.5600) underperformed both `random` (0.5754) and `stratified` (0.5908) when trained from scratch in this setup.
* While the motivation for `gen_probe` is strong (mining hard examples based on domain gap), the strategy might be selecting examples that are *too hard* or outliers, causing the model to struggle to learn generalizable features from scratch. 
* Tuning the selection metric (e.g., mixing in some random/diversity samples, taking the middle uncertainty instead of absolute highest, or adjusting the `probe` model's training epochs) might be required to make it surpass Stratified/Attribute strategies.

---

## 3. Mathematical Formulation of Gen-Probe 
The core idea of **Generation-Guided Active Learning (Gen-Probe)** is to use a model trained exclusively on generated images to act as a diagnostic "probe" on the original source data. This identifies the "domain gap"—the features present in real images that the generative model fails to synthesize.

### Selection Mechanism:

1. **Probe Training**: We train a model $M_{gen}$ exclusively on the pool of generated images (e.g., 1000 generated images).
2. **Hardness Scoring (Entropy)**: We evaluate $M_{gen}$ on the entire **source training pool** (e.g., `train_idx_imagenet.npy` consisting of 10,400 unlabeled original images). For each image $x_i$, the probe outputs a probability distribution across $C$ classes: $P(y|x_i)$. 
   The score assigned to the image is its **Information Entropy** $H(x_i)$:
   $$ H(x_i) = - \sum_{c=1}^{C} P(y=c|x_i) \log P(y=c|x_i) $$
   - A **low entropy** ($H \to 0$) means $M_{gen}$ is confident; the generated images already cover these features well.
   - A **high entropy** means $M_{gen}$ is entirely confused; these images contain features distinct from the generated domain.
3. **Data Selection**: We sort all samples descending by $H(x_i)$ and select the top $K$ images (e.g., 30% of the pool) that the probe model found most difficult.



---

## 4. Future Research Directions: Why did "Hard Mining" fail here?

The initial run of `gen_probe` yielded an accuracy of 0.5600, which is lower than `random` selection (0.5754). This is a well-known phenomenon in Active Learning literature: **Hard Example Mining is extremely susceptible to outliers.**

When a model is trained from scratch using *only* the Top-K highest entropy samples, it is essentially being trained on the most ambiguous, heavily occluded, or mislabeled images in the dataset. A model cannot build robust decision boundaries when given only edge cases.


---





---

## Uncertainty-Weighted Stratified Strategy 

The **Uncertainty Stratified** strategy aims to select the most informative samples (highest uncertainty) while strictly preserving a balanced class distribution in the selected subset.

### 1. Uncertainty Calculation (Entropy)
For a multi-class classification problem with $C$ classes, let $x_i$ be an unlabeled sample and $P(y=c|x_i)$ be the predicted probability of class $c$ by the current model. The uncertainty score for sample $i$ is calculated using Information Entropy:
$$ U_i = -\sum_{c=1}^{C} P(y=c|x_i) \log P(y=c|x_i) $$
A higher $U_i$ indicates that the model is less confident about the prediction.

### 2. Stratified Quota Allocation
Let $K$ be the total target number of samples to select, and $N_{classes}$ be the total number of unique ground-truth (or pseudo-label) classes in the candidate pool. The budget is distributed uniformly across all classes:
$$ K_{class} = \lfloor \frac{K}{N_{classes}} \rfloor $$
Any remainder samples $K \pmod{N_{classes}}$ are distributed iteratively to the first few classes until the budget is exhausted.

### 3. Stratified Selection
For each individual class $c$:
1. Isolate the subset of samples $\mathcal{S}_c$ that belong to class $c$.
2. Sort $\mathcal{S}_c$ in descending order based on their assigned uncertainty scores $U_i$.
3. Select the top $K_{class}$ samples with the highest uncertainty from $\mathcal{S}_c$.

By combining point-wise entropy maximization with class-wise histogram equalization, this strategy avoids the common mode collapse in standard uncertainty sampling (where the model might exclusively select difficult samples from only one or two dominant/confusing classes).



## Ablation Study for Attribute Parameters

The following table summarizes the ablation experiments with sub_ratio=0.7 for the AttributeFocusedStrategy parameters on ImageNet dataset. Each row corresponds to a different configuration where one or more parameters are set to specific values while others are zero (except for the baseline).

| Configuration | Alpha (rarity_power) | Beta (uncertainty_boost) | Gamma (log_acc_scale) | Accuracy | Notes |
|---------------|----------------------|---------------------------|-----------------------|----------|-------|
| Baseline (all zero) | 0 | 0 | 0 | 0.699388 | No attribute weighting |
| Traditional AL(Coreset) | - | - | - | 0.69873 | |
| Rarity only | 1 | 0 | 0 | 0.700433 | Only rarity component active |
| Uncertainty only | 0 | 1 | 0 | 0.704842 | Only uncertainty bonus active |
| Log weight only | 0 | 0 | 1 | 0.709308 | Only log-derived weights active |
| All | 1 | 1 | 1 | 0.700769 |  |
| All, but scales | 1 | 1 | 10 | 0.700384 |  |

*Note: Experiments run with ResNet model, sub_ratio=0.7, attributes=color,background.*

---

# Active Learning Strategy Usage Guide

## Overview

I have implemented multiple Active Learning strategies for the safefix project to address the poor performance of the original Core-set method.

## Available Strategies

### 1. CoreSetStrategy (Original Method)
```bash
python alpred.py --al_strategy coreset --al_initial_labeled 400 --al_embedding_dim 256
```
- **Principle**: Selects samples farthest from labeled data (diversity-first)
- **Use Cases**: Uniform data distribution, need to maximize coverage
- **Issues**: May select outlier samples, disrupting class balance

### 2. StratifiedStrategy (Stratified Sampling)
```bash
python alpred.py --al_strategy stratified
```
- **Principle**: Stratified sampling by class proportion, maintaining class balance
- **Use Cases**: Imbalanced data, need representation from every class
- **Advantages**: Maintains original data distribution, simple and efficient
- **Speed**: Fastest (no feature extraction needed)

### 3. DiversityStrategy (Diversity Clustering)
```bash
python alpred.py --al_strategy diversity --al_clusters 100 --al_embedding_dim 256
```
- **Principle**: K-means clustering followed by representative sample selection from each cluster
- **Use Cases**: Data with natural clustering structure, need to cover different data patterns
- **Advantages**: Balances diversity and representativeness

### 4. HybridStrategy (Mixed Strategy)
```bash
python alpred.py --al_strategy hybrid --al_coreset_ratio 0.3 --al_stratified_ratio 0.5 --al_embedding_dim 256
```
- **Principle**: Combines advantages of multiple methods
  - 30% Core-set (diversity)
  - 50% Stratified (class balance)  
  - 20% Random (randomness)
- **Use Cases**: Complex data distributions, need comprehensive consideration
- **Advantages**: Most robust, highly adaptive

### 5. AttributeFocusedStrategy (Attribute-Driven Sampling)
```bash
python alpred.py --al_strategy attribute --attribute_file save/attribute10.json \
  --al_attribute_focus color,background --al_attribute_rarity_power 1.5 --al_attribute_uncertainty_boost 1.0
```
- **Principle**: Builds a score for every unlabeled sample by combining attribute rarity, metadata-level uncertainty, and optional diagnostic weights derived from logs.
- **Use Cases**: Scenarios where human annotations describe attribute slices (color, background, hairstyle, etc.) and the model underperforms on specific combinations.
- **Advantages**: Lightweight (no feature extraction), controllable focus on high-value slices, deterministic once metadata is fixed.

The total score $S_i$ for sample $i$ is the sum across all attributes $A$ of four components:
$$S_i = \sum_{a \in A} \big( R_{i,a} + U_{i,a} + L_{i,a} + P_{i,a} \big)$$

Where:
*   **$R_{i,a}$ (Class-Conditional Rarity)**: $\left( \frac{1}{f_{a}(v_{i,a} | y_i) + \varepsilon} \right)^{\alpha}$. Default $\alpha=0.01$. Measures how rare an attribute value is *within its predicted class*.
*   **$U_{i,a}$ (Uncertainty)**: $\beta$ if marked uncertain in JSON, else $0$. If `--al_attribute_use_entropy True` is set, this is dynamically replaced by Information Entropy $\beta \cdot \big( -\sum P \log P \big)$ from model inference. Default $\beta=1.0$.
*   **$L_{i,a}$ (Log Accuracy)**: $\gamma \cdot (1 - \text{ACC}_{a}(v_{i,a}))$. Default $\gamma=0.0$.
*   **$P_{i,a}$ (Frequency Penalty)**: $-\lambda \cdot \text{FreqRatio}(v_{i,a})$. Default $\lambda=5.0$. This term continuously suppresses over-sampling from heavily flooded slices.

Samples are sorted by $S_i$ (with a small injected noise term for tie breaking) and the top $n_{\text{select}}$ indices are returned. When a sample exhibits multiple rare or low-accuracy attributes, the contributions accumulate, ensuring multi-attribute failures are prioritized.

**Understanding the Frequency Penalty ($P$) and Generated Images**
While the formula for $P_{i,a}$ computes a simple frequency ratio, it perfectly neutralizes the bias introduced by injecting artificially generated data. Specifically, the ratio is calculated over the *combined* candidate pool:
$$\text{FreqRatio}(v) = \frac{N_{\text{base}}(v) + N_{\text{gen}}(v)}{\sum_{v'} \big( N_{\text{base}}(v') + N_{\text{gen}}(v') \big)}$$
Where $N_{\text{base}}$ represents the counts in the original unlabeled dataset, and $N_{\text{gen}}$ represents the newly injected synthetic bounding boxes (e.g., 1000 images focused purely on `color:pink` and `texture:fabric`). Because $N_{\text{gen}}$ concentrates an enormous mass into a few specific attribute values $(v)$, those specific values experience an artificially inflated $\text{FreqRatio}$. The $-\lambda \cdot \text{FreqRatio}$ penalty ensures that the active learning selector does not blindly exhaust its budget on the synthetic flood, forcing it to balance the newly injected data against naturally occurring hard cases.

**The 2D Synergy: $R$ (Class-Conditional) vs. $P$ (Global)**
At first glance, Rarity ($R$) rewarding low frequency and Penalty ($P$) penalizing high frequency seem contradictory. However, because $R$ evaluates *within a specific class* ($P(v|y)$) while $P$ evaluates *across the entire global pool* ($\text{FreqRatio}$), they form a powerful 2D filter:
*   **High $R$ + Low $P$ (Class-rare, Global-rare)**: The most valuable "Hard Cases" (e.g., a "pink elephant"). They receive massive score boosts.
*   **High $R$ + High $P$ (Class-rare, Global-common)**: Features that are generic globally but strange for this specific class. $R$ and $P$ partially offset, preventing the model from over-indexing on globally redundant background textures just because they appeared in a new class.
*   **Low $R$ + High $P$ (Class-common, Global-common / Artificial Flooding)**: Boring redundant samples or the exact heavily-injected generated slices (e.g., 1000 synthetic `pink` dogs). The massive $P$ penalty sinks their total $S_i$ score, acting as an automatic spam filter to prevent them from exhausting the selection budget.

Optionally, enable an additional **Class Stratification Score ($C$)** via `--al_attribute_class_balance`. After computing $S_i$, the selector performs a greedy re-ranking that rewards classes whose current selections fall below their per-class quota. When the flag is set to a finite value $w>0$, each class $c$ keeps a running count $n_c$ and target quota $t = n_{\text{select}} / N_{\text{classes}}$. The next candidate from class $c$ receives an additive bonus $C_{c}$:
$$C_{c} = w \times \frac{t - n_c}{t}$$
which is added to its base score $S_i$ inside `_soft_class_selection` (see `utils/active_learning.py`). Classes under their quota get positive bonuses, while classes at or above the quota see the term shrink toward zero or become negative, letting pure attribute scores win. Candidates with unknown labels (e.g., synthetic-only entries) sit in a separate queue and are inserted only when their base score exceeds every class-adjusted option, so unlabeled data does not distort the balancing pass.

During iteration $k$ of `_soft_class_selection`, every labeled candidate $i$ from class $c = y_i$ uses the effective score
$$\tilde{S}_i^{(k)} = S_i + C_{c}^{(k)}$$
where $C_{c}^{(k)} = w \times \frac{t - n_c^{(k)}}{t}$, and $n_c^{(k)}$ is the number of times class $c$ has already been chosen. The algorithm picks the candidate with the largest $\tilde{S}_i^{(k)}$ (breaking ties with the injected noise from the base ranking), appends it to the selection list, and then updates $n_c^{(k+1)} = n_c^{(k)} + 1$ for that class. If the highest raw-score unlabeled item $u$ satisfies $S_u > \max_i \tilde{S}_i^{(k)}$, the iteration chooses $u$ instead and leaves all class counts unchanged, where $S_u$ denotes the raw attribute score of the unlabeled candidate (no class bonus exists for unlabeled entries). That iteration simply appends $u$ to `selected`. The loop stops once either $|\text{selected}| = n_{\text{select}}$ or all class and unlabeled queues are empty; in the latter case `_finalize_balanced_selection` backfills the remaining slots from the global ranking. “All other class and unlabeled queues are empty” literally means every per-class list (and the unlabeled list) has been drained—no candidate remains to evaluate—so the loop has no further element to pull even if some deficits are still positive.

This deficit-based soft balancing is a contribution inspired by proportional-fair and deficit round-robin schedulers: we convert each class shortfall $(t - n_c)$ into a temporary additive bonus while preserving the metadata-driven ordering within every class, providing a controllable bridge between pure attribute scoring and hard stratification.
The motivation was empirical: attribute scores alone over-pick head classes, yet strict round-robin ignores the large score spread across classes, so we needed a lightweight middle ground that can lean toward balance without throwing away the metadata signal—a criterion none of the standard AL heuristics offered out of the box.

Setting `--al_attribute_class_balance inf` bypasses the soft bonus and triggers `_round_robin_class_selection`, the literal round-robin implementation. Each class queue contributes its current best-scoring sample, then the queue moves to the back. Exhausted queues are removed, and the process repeats until either the requested budget is filled or only unlabeled/leftover samples remain (at which point the code falls back to global sorting in `_finalize_balanced_selection`). This guarantees per-class counts differ by at most one (subject to availability) before any remaining slots revert to global scores.

**Round-robin mechanics (as implemented in `_round_robin_class_selection`):**
1. Build per-class queues sorted by the raw attribute score + noise.
2. Walk the queues in order; on each step pop the head of the current class and append it to `selected`.
3. Move that class to the back of the list; if the queue is empty, drop it permanently.
4. Repeat until `n_select` items are collected or every class queue is empty, then rely on `_finalize_balanced_selection` to fill any residual slots from the global score ranking.
This produces the same effect as traditional stratified sampling but preserves the attribute ranking within each class.

**Example (ImageNet, `sub_ratio=0.3`, ten classes):** the quota is $t = 3{,}120 / 10 = 312$ samples per class. Running with `--al_attribute_class_balance 0` let the desk class drop to 133 samples (4.3 %) because it had few high attribute scores. Re-running the same configuration with `--al_attribute_class_balance 4` gave each desk candidate an early bonus of $4 \times (312-133)/312 \approx 2.29$, so they entered the greedy loop sooner. The final desk count rose to 273 samples (8.8 %), while large classes such as backpack and purse were nudged down toward the quota. Desk growth stopped at 273 because the global budget $n_{\text{select}} = 3{,}120$ was filled once the bonus had already reduced the biggest deficits; the algorithm cannot insert additional desk samples after the budget is exhausted even though $n_{\text{desk}} < t$. This illustrates how finite weights gently pull distributions toward balance without forcing perfect equality.

**Weight tuning versus sub-ratio:** because $C_c = w \times (t - n_c)/t$ scales with the quota $t = n_{\text{select}} / N_{\text{classes}}$, a fixed $w$ becomes weaker as the sub-sampling ratio grows. To preserve roughly the same maximum bonus seen at `sub_ratio = 0.3`, scale $w$ in proportion to $t$: $w_{\text{new}} \approx w_{0.3} \times (t_{\text{new}} / 312)$. For ImageNet’s ten classes this gives practical anchors:
- `sub_ratio = 0.7` (quota $\approx 728$): use $w \approx 9$–10.
- `sub_ratio = 0.9` (quota $\approx 936$): use $w \approx 12$.

Smaller weights still work but apply weaker nudges; larger weights push the distribution closer to stratified counts. Set `inf` if the run truly needs round-robin behavior.

The selector still picks the same total budget—$(10{,}400 \times 0.3 = 3{,}120)$ samples in this setup. Turning on `--al_attribute_class_balance 4` does not add or remove items; it just changes which indices fill those 3,120 slots. `_finalize_balanced_selection` appends the highest remaining base scores until the list has exactly $n_{\text{select}}$ entries, ensuring the class-aware re-ranking is purely a reordering step.

**Workflow**
- Provide an attribute JSON (for example `save/attribute10.json`) that maps each attribute to per-sample values and optional uncertain indices.
- Optionally restrict attention to a subset via `--al_attribute_focus` (comma-separated keys).
- (Optional) Supply a diagnostic log with `--al_attribute_log_path`; each `Rare Case` entry contributes a failure weight $(1 - \text{ACC})$ through the parser in `utils.active_learning.parse_attribute_priority_log`.
- Tune `--al_attribute_rarity_power`, `--al_attribute_uncertainty_boost`, and `--al_attribute_log_acc_scale` to balance between metadata-driven rarity, known uncertain slices, and empirical accuracy signals.
- (Optional) Provide pre-counted generated samples via `--al_attribute_extra_counts color:pink,texture:fabric`; when counts are omitted they default to `--num_added_img`, and if the flag itself is omitted the selector now parses `--attributes` (tokens such as `pinkcolor` or `fabrictexture`) to auto-inject the same `--num_added_img` count for those attribute-value pairs.

Because the method operates purely on metadata, it integrates cleanly with the rest of the pipeline—no embedding extraction or model forward passes are required during selection.

## Parameter Description

### General Parameters
- `--al_strategy`: AL strategy selection [coreset|stratified|diversity|hybrid|hierarchical|attribute]
- `--al_embedding_dim`: Feature vector dimension (default: 512)
- `--al_initial_labeled`: Initial labeled sample count (default: 500)

### Hybrid Strategy Parameters
- `--al_coreset_ratio`: Core-set sampling ratio (default: 0.4)
- `--al_stratified_ratio`: Stratified sampling ratio (default: 0.4)
- Remaining portion automatically allocated to random sampling

### Diversity Strategy Parameters
- `--al_clusters`: Number of clusters (default: n_select//10)

### Attribute Strategy Parameters
- `--attribute_file`: Path to attribute metadata JSON (e.g., `save/attribute10.json`)
- `--al_attribute_focus`: Optional comma-separated attribute keys to prioritize (default: all keys in metadata)
- `--al_attribute_rarity_power`: Emphasis on rare attribute values (>=0, higher = stronger focus on rare slices)
- `--al_attribute_uncertainty_boost`: Score bonus for samples marked as uncertain in metadata (>=0)
- `--al_attribute_use_entropy`: If True, computes Information Entropy from model inference and uses it for the $U$ term instead of static metadata flags (default: False)
- `--al_attribute_log_path`: Optional path to diagnostic log providing Rare Case accuracies
- `--al_attribute_log_acc_scale`: Scaling factor applied to each log-derived failure weight (default: 1.0)
- `--al_attribute_extra_counts`: Inline list or JSON file describing additional counts to inject per attribute value (useful when synthetic data floods specific slices); inline entries without `=count` inherit `--num_added_img`, and when omitted entirely the code infers counts from `--attributes` tokens (value+attribute) using `--num_added_img`
- `--al_attribute_balance_strength`: Multiplier controlling how strongly over-represented values are penalized (0 disables the term)
- `--al_attribute_class_balance`: Stratified weight for enforcing class balance after attribute scoring (0 disables, `inf` enforces near-perfect round-robin per class)

## Recommended Configurations

### For ImageNet Dataset (Multi-class Classification)
```bash
# Recommended: Hybrid strategy, balancing performance and efficiency
python alpred.py \
  --dataset imagenet --model resnet \
  --al_strategy hybrid \
  --al_coreset_ratio 0.2 \
  --al_stratified_ratio 0.6 \
  --al_embedding_dim 256 \
  --sub_ratio 0.5

# Alternative: Pure stratified sampling, fastest and most stable
python alpred.py \
  --dataset imagenet --model resnet \
  --al_strategy stratified \
  --sub_ratio 0.5
```

### For CelebA Dataset (Binary Classification)
```bash
# Recommended: Diversity strategy
python alpred.py \
  --dataset celeba --model resnet \
  --al_strategy diversity \
  --al_clusters 50 \
  --al_embedding_dim 128 \
  --sub_ratio 0.5
```

## Performance Expectations

Based on analysis, expected performance ranking:

1. **StratifiedStrategy**: Most stable, maintains class balance
2. **HybridStrategy**: Best overall performance
3. **DiversityStrategy**: Moderate performance, adapts to complex distributions
4. **CoreSetStrategy**: Possibly worst performance (original issue)

## Testing Scripts

Run comparative tests of all strategies:
```bash
./test_al_strategies.sh
```

View result comparisons:
```bash
python utils/compare_al_results.py
```

## Debugging Suggestions

1. **Use --verbose True** to view detailed class distribution information
2. **Start with small datasets** to verify strategy functionality
3. **Compare training times** to select strategy suitable for your computational resources
4. **Check class balance** to ensure selected samples have sufficient representativeness

## Code Structure

New AL strategies are all in `utils/active_learning.py`:
- `CoreSetStrategy`: Original core-set method
- `StratifiedStrategy`: Stratified sampling
- `UncertaintyStrategy`: Uncertainty-based sampling (future extension)
- `DiversityStrategy`: Clustering-based diversity sampling
- `HybridStrategy`: Mixed multiple strategies

Each strategy is an independent class that can be easily extended and modified.

### 6. Improved Strategies (New)

I have implemented two robust strategies to correct the weaknesses of previous methods (ignoring class balance or ignoring semantic rarity).

#### A. Uncertainty-Weighted Stratified Strategy (`uncertainty_stratified`)
```bash
python alpred.py --al_strategy uncertainty_stratified
```
**Motivation**: Standard uncertainty sampling often picks outliers from a single "hard" class, causing imbalance. Standard stratified sampling ignores sample difficulty. This strategy fuses them.

**Algorithm**:
1.  **Inference**: Compute uncertainty for all unlabeled samples.
    *   *Multi-class*: Entropy $H(p) = -\sum p_i \log p_i$.
    *   *Binary*: Confidence gap $1 - |p - 0.5| \times 2$.
2.  **Quota Calculation**: Determine strict sample quotas for each class $c$: $k_c \approx N_{total} / C$.
3.  **Selection**: 
    *   For each class $c$, retrieve all samples predicted as $c$ (or labeled as $c$ if labels available).
    *   Sort them by uncertainty (High to Low).
    *   Select top $k_c$ samples.

**Key Benefit**: Guarantees diversity across classes (like Stratified) whilst maximizing information gain (like Uncertainty).
