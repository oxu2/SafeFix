# Journal - oxu (Part 1)

> AI development session journal
> Started: 2026-02-24

---

## Session 37: Add Qwen3 train20k image-id filter for non-brown and non-rainy images

**Date**: 2026-04-15
**Task**: qwen3-train20k-non-brown-non-rainy-image-ids

### Summary

The user requested a Python file to find the Qwen3 COCO pose train20k training-set ids where `clothes_color != brown` and `weather != rainy`. After the user clarified that `idx` means `image_id` rather than CSV row index, added a dedicated script under `scripts/` that reads the canonical Qwen3 train20k CSV, writes the matched `image_id` values to a text file, and also writes a metadata CSV with both `image_id` and reference `row_idx`.

### Main Findings

- The canonical input CSV used by the new script is:
  - `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv`
- New script:
  - `scripts/find_qwen3_train20k_non_brown_non_rainy_indices.py`
- The script now treats the user-requested `idx` as:
  - `image_id`
- The script still records a 0-based `row_idx` in the metadata CSV for debugging / traceability, but the main text output is one `image_id` per line.
- Running the script on the canonical Qwen3 train20k CSV produced:
  - `18849` matched image ids
- First matched `image_id` values:
  - `36`
  - `49`
  - `86`
  - `110`
  - `113`
  - `136`
  - `165`
  - `192`

### Main Changes

- Added:
  - `scripts/find_qwen3_train20k_non_brown_non_rainy_indices.py`
- Generated output artifacts:
  - `save/qwen3_train20k_non_brown_non_rainy_image_ids.txt`
  - `save/qwen3_train20k_non_brown_non_rainy_matches.csv`

### Validation

- Ran:
  - `python3 scripts/find_qwen3_train20k_non_brown_non_rainy_indices.py`
- Confirmed generated outputs exist and contain data:
  - `wc -l save/qwen3_train20k_non_brown_non_rainy_image_ids.txt`
    - `18849`
  - `wc -l save/qwen3_train20k_non_brown_non_rainy_matches.csv`
    - `18850`
- Confirmed the text file contains raw `image_id` values rather than row indices.
- Confirmed the metadata CSV contains:
  - `row_idx`
  - `image_id`
  - `clothes_color`
  - `weather`

### Status

[OK] **Completed**

---

## Session 36: Summarize Qwen3 rare-threshold sweep into report and journal

**Date**: 2026-04-15
**Task**: qwen3-rare-threshold-sweep-reporting

### Summary

The user asked to wait for the Qwen3 `rare_label_threshold=0.97 / 0.95 / 0.93` post-hoc analyses to finish, then summarize the result into `report.md` and record it in the Trellis journal. Reused the completed Qwen3 GPU `0` reproducibility rerun artifacts, ran the three post-hoc slice analyses, compared the outputs across thresholds, and documented the stable rare-case sets for both `val` and `test`.

### Main Findings

- The Qwen3 post-hoc rare-threshold sweep is complete for the April 15 reproducibility rerun:
  - run root:
    - `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro`
  - generated slice-analysis roots:
    - `slice_analysis/qwen3_rare0p97/`
    - `slice_analysis/qwen3_rare0p95/`
    - `slice_analysis/qwen3_rare0p93/`
- The three Qwen3 `val_slice_stats.csv` files are byte-identical across thresholds.
- The three Qwen3 `test_slice_stats.csv` files are also byte-identical across thresholds.
- Interpretation:
  - for this checkpoint, lowering `rare_label_threshold` from `0.97` to `0.95` or `0.93` did not change Qwen3 slice rankings or rare/hard-case labels
- Stable Qwen3 **val** rare cases across all three thresholds:
  - `pose=lying`: `AP=0.3623`, `27` images
  - `time_of_day=night`: `AP=0.5826`, `61` images
  - `clothes_color=brown`: `AP=0.5910`, `37` images
  - `time_of_day=morning`: `AP=0.6135`, `105` images
  - `time_of_day=evening`: `AP=0.6252`, `161` images
  - `clothes_color=gray`: `AP=0.6336`, `113` images
- Stable Qwen3 **test** rare cases across all three thresholds:
  - `pose=lying`: `AP=0.3906`, `34` images
  - `time_of_day=evening`: `AP=0.5737`, `164` images
  - `weather=rainy`: `AP=0.5763`, `31` images
  - `time_of_day=night`: `AP=0.5802`, `74` images
  - `background_style=rural`: `AP=0.6045`, `86` images
  - `clothes_color=pink`: `AP=0.6054`, `43` images
  - `clothes_color=gray`: `AP=0.6147`, `119` images
  - `clothes_color=brown`: `AP=0.6202`, `38` images

### Main Changes

- Updated `report.md` under:
  - `04/17/2026 Update — GPU 0 and GPU 3 Pose-Training Runs Finished`
- Added a new Qwen3 sweep note covering:
  - the completed `0.97 / 0.95 / 0.93` analysis roots
  - the identical-output conclusion across thresholds
  - the stable `val` and `test` rare-case lists
- Updated `.handoff.md` so the current baton explicitly includes the completed Qwen3 sweep and its invariant-threshold result.

### Validation

- Verified the output files exist under:
  - `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro/slice_analysis/`
- Verified the Qwen3 sweep outputs are identical across thresholds with:
  - matching SHA-256 hashes for the three `val_slice_stats.csv` files
  - matching SHA-256 hashes for the three `test_slice_stats.csv` files
- Extracted all rows marked `rare_case` from the three Qwen3 `val` CSVs and confirmed the sets are identical.
- Extracted all rows marked `rare_case` from the three Qwen3 `test` CSVs and confirmed the sets are identical.
- Re-read the updated 4/17 addendum in `report.md` and confirmed the new Qwen3 sweep note is appended without overwriting the earlier summary table.

### Status

[OK] **Completed**

---

## Session 35: Finalize 4/17 report addendum for later Qwen3 / InternVL3 reruns

**Date**: 2026-04-15
**Task**: report-0417-followup-rerun-addendum

### Summary

The user asked whether the later April 15 reruns had finished and requested that the new results be added into the existing `04/17/2026` report section without overwriting the original summary table. Checked the queued InternVL3 GPU `3` training-only rerun plus its post-hoc rare-threshold watcher, discovered the later Qwen3 reproducibility rerun on GPU `0`, extracted the final metrics and slice-analysis outputs, and appended an additive comparison block to `report.md`.

### Main Findings

- The InternVL3 GPU `3` training-only rerun is complete:
  - training log:
    - `log_pose/pose_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404.log`
  - output dir:
    - `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404`
  - best epoch:
    - `0`
  - held-out metrics:
    - val `AP=0.6412`
    - test `AP=0.6371`
  - compared with the earlier 4/17 InternVL3 bs8 run:
    - val `+0.0011`
    - test `+0.0014`
    - `pose=lying` improved from `0.3625` to `0.4021`
- The queued InternVL3 post-hoc rare-threshold sweep is also complete:
  - watcher log:
    - `log_pose/posthoc_slice_analysis_internvl3_trainonly_gpu3_20260415_183404.log`
  - generated slice-analysis roots:
    - `slice_analysis/internvl3_rare0p97/`
    - `slice_analysis/internvl3_rare0p95/`
    - `slice_analysis/internvl3_rare0p93/`
  - both the `val_slice_stats.csv` files and the `test_slice_stats.csv` files are byte-identical across all three thresholds
  - interpretation:
    - for this checkpoint, lowering `rare_label_threshold` from `0.97` to `0.95` or `0.93` did not change slice rankings or rare/hard-case labels
- The later Qwen3 reproducibility rerun is complete:
  - log:
    - `log_pose/pose_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro.log`
  - output dir:
    - `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro`
  - best epoch:
    - `0`
  - held-out metrics:
    - val `AP=0.6407`
    - test `AP=0.6341`
  - compared with the earlier 4/17 Qwen3 bs8 run:
    - val `-0.0038`
    - test `-0.0011`
    - `pose=lying` improved from `0.3091` to `0.3906`
- The original 4/17 top summary remains preserved in `report.md`; the new content is appended as a later addendum instead of replacing the earlier table.

### Main Changes

- Updated `report.md` under:
  - `04/17/2026 Update — GPU 0 and GPU 3 Pose-Training Runs Finished`
- Added an additive follow-up rerun comparison covering:
  - Qwen3 GPU `0` repro vs earlier Qwen3 GPU `3` run
  - InternVL3 GPU `3` train-only rerun vs earlier InternVL3 GPU `0` run
  - the InternVL3 `rare_label_threshold` sweep result at `0.97`, `0.95`, and `0.93`
- Updated `.handoff.md` so the baton no longer describes the InternVL3 GPU `3` rerun as active and now includes the later Qwen3 repro plus the sweep-completion result.

### Validation

- Verified the old tracked InternVL3 rerun PIDs are no longer alive:
  - `ps -fp 1999088`
  - `ps -fp 2001414`
- Re-read the final metrics from:
  - `log_pose/pose_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404.log`
  - `log_pose/pose_train20k_qwen3_bs8_e2_rare1p1_gpu0_repro.log`
- Verified the InternVL3 post-hoc sweep completed all requested thresholds by checking:
  - `log_pose/posthoc_slice_analysis_internvl3_trainonly_gpu3_20260415_183404.log`
- Verified the sweep outputs are identical across thresholds with:
  - matching SHA-256 hashes for the three `val_slice_stats.csv` files
  - matching SHA-256 hashes for the three `test_slice_stats.csv` files
- Re-read the updated top of `report.md` and confirmed the original summary table is still present above the new addendum.

### Status

[OK] **Completed**

---

## Session 34: Launch InternVL3 GPU 3 training-only rerun for post-hoc rare-threshold sweep

**Date**: 2026-04-15
**Task**: pose-train-internvl3-gpu3-trainonly-rare-sweep

### Summary

The user requested a fresh InternVL3 pose rerun on GPU `3`, but explicitly rejected the current default `rare_label_threshold=1.1`. Confirmed that the rare threshold only affects post-train slice analysis rather than the training loop itself, so switched the workflow to a training-only rerun on GPU `3` and deferred slice analysis until after training finishes. The requested follow-up is to run separate stat-only analyses at thresholds `0.97`, `0.95`, and `0.93`.

### Main Findings

- Confirmed the current training script default is still:
  - `train_pose_estimation.py`
  - `--analysis-rare-label-threshold 1.1`
- Because the user does not want `1.1`, the new active run disables inline analysis entirely:
  - `--analyze-after-train false`
- Initial `nohup`-only background launch attempts on GPU `3` did not persist after the shell exited.
- The successful detached launch uses `setsid` plus the `ad` environment Python binary directly.

### Active Run

- Current active InternVL3 GPU `3` training-only rerun:
  - PID: `1999088`
  - output dir:
    - `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404`
  - log:
    - `log_pose/pose_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404.log`
  - pid file:
    - `log_pose/pose_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404.pid`
  - command shape:
    - `CUDA_VISIBLE_DEVICES=3 setsid nohup /people/cs/o/oxx220000/.conda/envs/ad/bin/python -u train_pose_estimation.py train --assigner internvl3 --analyze-after-train false --output-dir log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404`
- Queued post-train sweep watcher:
  - PID: `2001414`
  - script:
    - `scripts/run_internvl3_posthoc_rare_sweep.sh`
  - log:
    - `log_pose/posthoc_slice_analysis_internvl3_trainonly_gpu3_20260415_183404.log`
  - pid file:
    - `log_pose/posthoc_slice_analysis_internvl3_trainonly_gpu3_20260415_183404.pid`
  - requested thresholds:
    - `0.97`
    - `0.95`
    - `0.93`

### Validation

- Verified the new process is alive:
  - `ps -fp 1999088`
- Verified the queued watcher is alive:
  - `ps -fp 2001414`
- Verified early log startup:
  - `Current time: 0415183413`
  - `Training start timestamp: 2026-04-15 18:34:13`
  - `Dataset sizes after split: train=20000, val=1173, test=1173`
- Verified early training progress:
  - `[Epoch 0] step 50/2500: ...`
  - `[Epoch 0] step 100/2500: ...`
- Verified GPU binding:
  - `nvidia-smi` showed PID `1999088` on GPU `3`
  - GPU `3` memory rose to about `22702 MiB`

### Next Steps

- Wait for the GPU `3` training-only rerun to finish; the watcher will then run the queued stat-only analyses automatically.
- Review the generated `val` and `test` slice-analysis outputs under:
  - `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_trainonly_gpu3_20260415_183404/slice_analysis/`
- `report.md` was intentionally not updated yet because no new stable end-of-run metrics exist.

### Status

[IN PROGRESS] **GPU 3 training-only rerun is active; post-hoc threshold sweep pending**

---

## Session 33: Compare `time_of_day` instability between Qwen3 and InternVL3 pose-slice logs

**Date**: 2026-04-15
**Task**: compare-evening-night-pose-logs

### Summary

Read the two requested top-level pose-training logs and then checked the underlying Qwen3 / InternVL3 Gemini-corpus CSV annotations to see whether `time_of_day=evening` and `time_of_day=night` should be treated as separate stable concepts. The main result is that the raw four-way `time_of_day` labels are fairly unstable across assigners, while the merged `evening|night` low-light bucket is much more consistent.

### Main Findings

- Confirmed the current bs8 pose logs surface different low-light slices on the held-out test split:
  - Qwen3 log `log_pose/pose_train20k_qwen3_bs8_e2_rare1p1_20260415_144429.log`
    - `time_of_day=evening`: `AP=0.5629`, `164` images
    - `time_of_day=night`: `AP=0.5816`, `74` images
  - InternVL3 log `log_pose/pose_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530.log`
    - `time_of_day=night`: `AP=0.5725`, `191` images
    - `time_of_day=evening`: `AP=0.6088`, `72` images
- Verified the corpus still defines four distinct `time_of_day` words:
  - `morning`
  - `afternoon`
  - `evening`
  - `night`
- Compared raw `time_of_day` distributions in the underlying annotation CSVs:
  - `train20k`
    - Qwen3: `afternoon=14130`, `morning=2134`, `evening=2660`, `night=1076`
    - InternVL3: `afternoon=10293`, `morning=5607`, `evening=1195`, `night=2905`
  - `val2017`
    - Qwen3: `afternoon=1659`, `morning=227`, `evening=325`, `night=135`
    - InternVL3: `afternoon=1193`, `morning=652`, `evening=152`, `night=349`
- The split between `evening` and `night` is assigner-dependent, but the merged low-light share is much closer:
  - `train20k`
    - Qwen3 `evening+night`: `3736 / 20000 = 18.68%`
    - InternVL3 `evening+night`: `4100 / 20000 = 20.50%`
  - `val2017`
    - Qwen3 `evening+night`: `460 / 2346 = 19.61%`
    - InternVL3 `evening+night`: `501 / 2346 = 21.36%`
- Cross-assigner agreement numbers:
  - exact four-way `time_of_day` agreement
    - `val2017`: `1500 / 2346 = 63.94%`
    - `train20k`: `13005 / 20000 = 65.02%`
  - agreement after merging `evening` and `night` into one low-light bucket
    - `val2017`: `2065 / 2346 = 88.02%`
    - `train20k`: `17664 / 20000 = 88.32%`
- The strongest direct evidence of boundary instability is that many Qwen3 `evening` labels map to InternVL3 `night`:
  - `val2017`: `117` images
  - `train20k`: `913` images

### Interpretation

- Semantically, `evening` and `night` are not identical concepts.
- Operationally in these VLM-assigned COCO pose annotations, they behave like a noisy split inside a broader low-light bucket.
- For slice-analysis stability and cross-assigner comparison, a merged bucket such as `low_light = evening | night` is better.
- For prompt-level control or generation semantics, keep the raw labels and optionally report both the raw and merged views.

### Validation

- Re-read the relevant log sections for both requested files.
- Recomputed `time_of_day` distributions directly from:
  - `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv`
  - `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`
  - `save/attribute_coco_pose_gemini_qwen3_8B.csv`
  - `save/attribute_coco_pose_gemini_internvl3_8B.csv`

### Status

[OK] **Completed**

---

## Session 32: Correct 04/17 report pose-training runtimes from top-level logs

**Date**: 2026-04-15
**Task**: report-0417-pose-runtime-correction

### Summary

Read the `04/17/2026` section at the top of `report.md`, checked the two requested top-level `log_pose/` training logs, and corrected the bs8 pose-training runtime values to match the explicit `Training elapsed` summaries in those logs. Also recorded the paired end-to-end wall-clock times in `report.md` and `.handoff.md` so later agents do not reuse the earlier shorter timings.

### Main Changes

- Verified the requested top-level logs contain explicit timing summaries near the end of each file:
  - `log_pose/pose_train20k_qwen3_bs8_e2_rare1p1_20260415_144429.log`
    - `Total training time for 2 epochs: 5358.44 seconds`
    - `Training elapsed (HH:MM:SS): 01:29:18`
    - `End-to-end elapsed (HH:MM:SS): 01:29:26`
  - `log_pose/pose_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530.log`
    - `Total training time for 2 epochs: 5619.12 seconds`
    - `Training elapsed (HH:MM:SS): 01:33:39`
    - `End-to-end elapsed (HH:MM:SS): 01:33:46`
- Updated the `04/17/2026 Update — GPU 0 and GPU 3 Pose-Training Runs Finished` table in `report.md`:
  - Qwen3 `Train Time`: `01:23:59` -> `01:29:18`
  - InternVL3 `Train Time`: `01:28:17` -> `01:33:39`
- Added an explicit timing note below that table with both training-only and end-to-end elapsed durations for each run.
- Updated `.handoff.md` so the current baton includes the corrected runtime values for both bs8 pose runs.

### Validation

- Re-read the edited `report.md` header section and confirmed the corrected times appear in the summary table.
- Re-read the timing note in `report.md` and confirmed it matches the extracted log values.
- Re-read `.handoff.md` and confirmed both runs now include training and end-to-end runtime bullets.

### Status

[OK] **Completed**

---

## Session 31: Run BLIP base-corpus COCO pose `val` assignment with shared-GPU-safe VQA batching

**Date**: 2026-04-15
**Task**: blip-coco-pose-base-val

### Summary

Read `SLACK_WEBHOOK_HANDOFF.md`, confirmed the correct BLIP entrypoint is `utils/blip.py`, then ran BLIP attribute assignment on the COCO pose `val` split using the base corpus `save/corpus_coco_pose.json`. The first full attempt failed on the first VQA-style attribute because `utils/blipFeature.py` hardcoded BLIP VQA batch size `128`, which was too large under current shared-GPU pressure. Added environment-controlled BLIP batch sizes, reran with `BLIP_VQA_BATCH_SIZE=32`, and completed the requested `val` artifact generation successfully.

### Main Changes

- Confirmed `SLACK_WEBHOOK_HANDOFF.md` is only the reusable Slack-post pattern; it does not change the BLIP invocation itself.
- Confirmed the correct script path is:
  - `python -u utils/blip.py`
  - there is no top-level `blip.py` in this repo
- Confirmed the requested base-corpus `val` run should explicitly use:
  - `--dataset coco_pose`
  - `--split val`
  - `--corpus save/corpus_coco_pose.json`
- Avoided overwriting older full-run artifacts by using explicit split-scoped outputs:
  - `save/attribute_coco_pose_blip_val.json`
  - `save/attribute_coco_pose_blip_val.csv`
  - `save/labels_coco_pose_blip_val.npy`
  - `save/label_coco_pose_blip_val.npy`
  - `save/train_idx_coco_pose_blip_val.npy`
  - `save/val_idx_coco_pose_blip_val.npy`
- Diagnosed the first full-run failure:
  - the run reached `Match is_arm_crossing`
  - then failed in `utils.blipFeature.question_answering(...)`
  - exception:
    - `torch.cuda.OutOfMemoryError`
  - the visible-device process reported the OOM as `GPU 0` because the run was started with `CUDA_VISIBLE_DEVICES=2`
- Updated `utils/blipFeature.py` to support environment-controlled batch sizes while preserving the previous defaults:
  - `BLIP_IMAGE_FEATURE_BATCH_SIZE` default `50`
  - `BLIP_CAPTION_BATCH_SIZE` default `100`
  - `BLIP_SIMILARITY_BATCH_SIZE` default `128`
  - `BLIP_VQA_BATCH_SIZE` default `128`
- Retried the requested full run on GPU `2` with:
  - `BLIP_VQA_BATCH_SIZE=32`
  - `CUDA_VISIBLE_DEVICES=2`
- The retry completed successfully and wrote the requested base-corpus `val` outputs.

### Validation

- [OK] `python3 -m py_compile utils/blipFeature.py utils/blip.py`
- [OK] A 1-image smoke run with the new override completed successfully:
  - command used `BLIP_VQA_BATCH_SIZE=32`
  - outputs:
    - `/tmp/attribute_coco_pose_blip_val_probe_vqa32.json`
    - `/tmp/attribute_coco_pose_blip_val_probe_vqa32.csv`
- [OK] Final output files now exist:
  - `save/attribute_coco_pose_blip_val.json`
  - `save/attribute_coco_pose_blip_val.csv`
  - `save/labels_coco_pose_blip_val.npy`
  - `save/label_coco_pose_blip_val.npy`
  - `save/train_idx_coco_pose_blip_val.npy`
  - `save/val_idx_coco_pose_blip_val.npy`
- [OK] `wc -l save/attribute_coco_pose_blip_val.csv` returned `2347`, confirming `2346` data rows plus header.
- [OK] Verified every attribute in `save/attribute_coco_pose_blip_val.json` has `2346` entries:
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

### Important Notes

- The new BLIP batch-size environment variables preserve prior behavior unless explicitly set.
- Under shared-GPU conditions, the highest-risk path is BLIP VQA / binary attributes, not the text-similarity attributes.
- For future shared-GPU BLIP runs, start with:
  - `BLIP_VQA_BATCH_SIZE=32`
- `report.md` was intentionally not updated for this run because this is an operational artifact for the base COCO pose corpus rather than a report-level result currently used in the main narrative.

### Status

[OK] **Completed**

---

## Session 30: Archive top-level `0/1`-line `log_pose` logs into `log_pose/NA`

**Date**: 2026-04-15
**Task**: archive-empty-one-line-log-pose-files

### Summary

Moved the remaining top-level `log_pose/` files that were effectively empty wrappers or shell stubs into `log_pose/NA/`. The rule used here was practical rather than semantic: if a top-level historical log had `0` lines or only `1` line, it was treated as an archive candidate and removed from the main `log_pose/` surface.

### Main Changes

- Archived empty refresh-shell logs such as:
  - `assign_attribute_coco_pose_internvl3_8B_gemini_refresh_gpu3_20260412_160214.log`
  - `assign_attribute_coco_pose_internvl3_8B_gemini_refresh_gpu3_20260412_160250.log`
  - `assign_attribute_coco_pose_qwen3_8B_gemini_refresh_gpu0_20260412_160214.log`
  - `assign_attribute_coco_pose_qwen3_8B_gemini_refresh_gpu0_20260412_160250.log`
- Archived one-line wrapper / stub logs such as:
  - `assign_attribute_coco_pose_qwen3_8B_all.log`
  - `assign_attribute_coco_pose_internvl3_8B_all.log`
  - `assign_attribute_coco_pose_qwen3_8B_gemini_train20k_only_gpu0_20260412_162135.log`
  - `assign_attribute_coco_pose_internvl3_8B_gemini_train20k_only_gpu3_20260412_162238.log`
  - `qwen3_val_then_train_after20k_gpu1_20260414_134529.log`
- Archived other empty / trivial top-level probe logs and their matching pid files when present, including:
  - `bg_probe_20260412_160746.*`
  - `bg_conda_probe_20260412_160831.*`
  - `blip_coco_pose_gemini_gpu1_20260410_064531.*`
  - `pose_coco_pretrained_gpu3_ad_20260403_034701.*`
  - `rtmpose_tiny_coco_gpu3_20260403_131338.*`
  - `test_20260413_131218.log`
  - `test_20260413_131231.log`
  - `test_20260413_131235.log`
  - `test_20260413_131243.log`
  - `test_20260413_131247.log`
  - `test_20260413_131304.log`

### Validation

- Checked top-level `log_pose/` line counts before the move and identified the `0/1`-line candidates.
- Confirmed those files now exist under `log_pose/NA/`.
- Confirmed `log_pose/NA/` remains ignored by Git.
- Updated older journal / handoff references where they still pointed at the old top-level locations of archived one-line logs.

### Status

[OK] **Completed**

---

## Session 29: Move failed traceback `assign_attribute` logs into `log_pose/NA`

**Date**: 2026-04-15
**Task**: archive-traceback-assign-logs

### Summary

Moved the failed `assign_attribute_coco_pose_*` logs that contain `Traceback (most recent call last)` into `log_pose/NA/` so those obsolete failure logs stay grouped with the earlier `timeNA` archive instead of remaining mixed into the current top-level `log_pose/` log list.

### Main Changes

- Moved these failed Qwen3 logs into `log_pose/NA/`:
  - `assign_attribute_coco_pose_qwen3_8B_gemini_gpu3_20260410_192913.log`
  - `assign_attribute_coco_pose_qwen3_8B_gemini_gpu3_20260410_193009.log`
  - `assign_attribute_coco_pose_qwen3_8B_gemini_train20k_only_gpu0_20260412_162238.log`
  - `assign_attribute_coco_pose_qwen3_8B_gemini_train20k_resume_gpu0_20260413_232901.log`
  - `assign_attribute_coco_pose_qwen3_8B_gemini_train20k_resume_gpu0_20260413_232934.log`
  - `assign_attribute_coco_pose_qwen3_8B_gemini_train20k_resume_gpu0_20260413_233109.log`
- Moved these failed InternVL3 logs into `log_pose/NA/`:
  - `assign_attribute_coco_pose_internvl3_8B_gemini_gpu2_20260410_192913.log`
  - `assign_attribute_coco_pose_internvl3_8B_gemini_gpu2_20260410_193009.log`
  - `assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu1_20260413_232901.log`
  - `assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu1_20260413_232917.log`
  - `assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu1_20260413_233109.log`
- Also moved the matching `.pid` files when they existed, so the archived failed launches remain bundled with their logs.

### Scope Note

- Did **not** move unrelated non-`assign_attribute` traceback logs such as:
  - `log_pose/pose_coco_pretrained_gpu3_ad_live.log`
- That file is a different training workflow and not part of the `unknown` / `timeNA` archive path the user asked to consolidate.

### Validation

- Verified the moved traceback logs now exist under `log_pose/NA/`.
- Verified `git status --ignored` continues to show `log_pose/NA/` as ignored.
- Updated older journal and handoff references where they previously pointed at top-level failed logs that have now been archived under `log_pose/NA/`.

### Status

[OK] **Completed**

---

## Session 28: Relocate archived `unknown` / `timeNA` artifacts into `log_pose/NA`

**Date**: 2026-04-15
**Task**: relocate-unknown-timeNA-archives

### Summary

Moved the previously archived `timeNA` artifacts that came from the pre-cleanup `time_of_day=unknown` runs into `log_pose/NA/` so they stay preserved in one place and remain outside Git tracking.

### Main Changes

- Created the archive folder:
  - `log_pose/NA/`
- Moved the old archived Qwen3 files into `log_pose/NA/`:
  - `attribute_coco_pose_gemini_train20k_qwen3_8B_timeNA.json`
  - `attribute_coco_pose_gemini_train20k_qwen3_8B_timeNA.csv`
  - `assign_attribute_coco_pose_qwen3_8B_gemini_train20k_restart_gpu0_20260413_131326_timeNA.log`
  - `assign_attribute_coco_pose_qwen3_8B_gemini_train20k_restart_gpu0_20260413_131326_timeNA.pid`
- Moved the old archived InternVL3 files into `log_pose/NA/`:
  - `attribute_coco_pose_gemini_train20k_internvl3_8B_timeNA.json`
  - `attribute_coco_pose_gemini_train20k_internvl3_8B_timeNA.csv`
  - `attribute_coco_pose_gemini_train20k_internvl3_8B_timeNA_partial.json`
  - `assign_attribute_coco_pose_internvl3_8B_gemini_train20k_restart_gpu1_20260413_131327_timeNA.log`
  - `assign_attribute_coco_pose_internvl3_8B_gemini_train20k_restart_gpu1_20260413_131327_timeNA.pid`
- Updated `.handoff.md` and the older journal section so future agents use the new archive location instead of the old mixed `save/` plus `log_pose/` paths.

### Ignore Behavior

- Added an explicit `.gitignore` entry for:
  - `log_pose/NA/`
- This is slightly redundant because `log_pose/*` already ignores that directory, but the explicit rule makes the intent clearer:
  - archived `unknown` / `timeNA` files should not be tracked by Git

### Validation

- Verified the moved files now exist under `log_pose/NA/`.
- Verified `git status --ignored` shows:
  - `log_pose/NA/`
  as ignored.

### Status

[OK] **Completed**

---

## Session 27: Adjust `.gitignore` to keep small log files trackable

**Date**: 2026-04-15
**Task**: gitignore-log-vs-artifact-policy

### Summary

Updated `.gitignore` so top-level text log files in `log_pose/` are no longer blanket-ignored, while checkpoints and other experiment artifacts under `log_pose/` remain ignored. Left the separate `logs/` directory ignored because it already contains several large legacy log files, and Git ignore rules cannot filter by size.

### Why This Change Was Made

- The previous ignore rules used:
  - `log_pose/`
  - `logs/`
- That hid everything under those paths from Git, including small top-level `.log` files that are useful for reproducibility and report provenance.
- At the same time, the actual heavy artifacts that should stay ignored are mainly:
  - checkpoints such as `*.pth` and `*.pt`
  - experiment subdirectories under `log_pose/`
  - prediction / metric / slice-analysis outputs inside those experiment subdirectories
- Git ignore rules cannot express a file-size threshold, so the practical policy was changed to path-based tracking for logs.

### Main Changes

- Replaced the blanket ignore for `log_pose/` with:
  - `log_pose/*`
  - `!log_pose/*.log`
- Result:
  - top-level `.log` files such as:
    - `log_pose/pose_train20k_qwen3_bs8_e2_rare1p1_20260415_144429.log`
    - `log_pose/pose_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530.log`
    are now visible to Git
  - experiment directories like:
    - `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_20260415_144429/`
    - `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/`
    remain ignored as directories, so their checkpoints and JSON artifacts stay ignored
  - the separate `logs/` directory remains ignored to avoid surfacing existing multi-megabyte legacy logs by default

### Validation

- `git check-ignore -v` confirms:
  - `log_pose/*.log` is matched by the negation rule and is no longer ignored
  - files under experiment artifact directories such as:
    - `best.pth`
    - `test_metrics.json`
    are still ignored through `log_pose/*`
- Checked the two April 15 bs8 pose logs:
  - Qwen3 log size: `40,130` bytes
  - InternVL3 log size: `40,864` bytes

### Status

[OK] **Completed**

---

## Session 26: Verify completed bs8 pose runs on GPU 0 and 3 and refresh report

**Date**: 2026-04-15
**Task**: pose-bs8-completion-verification-report-refresh

### Summary

Verified that the April 15 bs8 pose-training runs on GPU `3` (Qwen3) and GPU `0` (InternVL3) both finished successfully, extracted their final held-out metrics and main slice-level failure cases, and added a new top entry to `report.md` with the requested `04/17` label.

### Main Verification

- Checked the original launched PIDs:
  - Qwen3 on GPU `3`: `1623280`
  - InternVL3 on GPU `0`: `1624937`
- Confirmed both PIDs are no longer alive via `ps -fp 1623280,1624937`.
- Confirmed both training logs end with `Training complete. Artifacts: ...`.
- Confirmed both artifact roots contain:
  - `best.pth`
  - `last.pth`
  - `history.json`
  - `val_metrics_best.json`
  - `test_metrics.json`
  - prediction JSONs
  - `slice_analysis/.../{val,test}_slice_stats.{json,csv}`
- Because these two jobs were launched before the compact-checkpoint change, they still wrote the old full-size `best.pth` and `last.pth` files (about `451 MB` each).

### Final Metrics

- Qwen3 bs8 run on GPU `3`
  - artifact root:
    - `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_20260415_144429`
  - log:
    - `log_pose/pose_train20k_qwen3_bs8_e2_rare1p1_20260415_144429.log`
  - best epoch:
    - `0`
  - held-out val metrics:
    - `AP=0.6445`, `AP50=0.8734`, `AP75=0.7038`, `AR=0.7128`
  - held-out test metrics:
    - `AP=0.6352`, `AP50=0.8656`, `AP75=0.6902`, `AR=0.6993`
  - train-only time from `history.json`:
    - `5038.94 s` (`01:23:59`)
  - strongest test failure slices:
    - `pose=lying`: `AP=0.3091`, `ap_delta=-0.3261`, `34` images
    - `pose=sitting`: `AP=0.5503`, `ap_delta=-0.0849`, `329` images

- InternVL3 bs8 run on GPU `0`
  - artifact root:
    - `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530`
  - log:
    - `log_pose/pose_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530.log`
  - best epoch:
    - `0`
  - held-out val metrics:
    - `AP=0.6401`, `AP50=0.8622`, `AP75=0.7003`, `AR=0.7102`
  - held-out test metrics:
    - `AP=0.6357`, `AP50=0.8618`, `AP75=0.6936`, `AR=0.7016`
  - train-only time from `history.json`:
    - `5296.96 s` (`01:28:17`)
  - strongest test failure slices:
    - `pose=lying`: `AP=0.3625`, `ap_delta=-0.2733`, `29` images
    - `pose=sitting`: `AP=0.5480`, `ap_delta=-0.0877`, `288` images

### Report Update

- Added a new top section to `report.md`:
  - `04/17/2026 Update — GPU 0 and GPU 3 Pose-Training Runs Finished`
- The report section records:
  - completion status
  - held-out val/test metrics
  - best epoch
  - train-time totals
  - primary artifact roots and logs
  - the leading hard slices from the held-out test split
- Important date note:
  - the report heading uses `04/17/2026` because that was requested by the user
  - the underlying run logs and artifact timestamps remain `2026-04-15`

### Comparison Note

- These bs8 runs are **not** directly configuration-matched to the earlier `4/12` probes because the newer runs use:
  - `batch_size=8`
  - `analysis_rare_label_threshold=1.1`
- Relative to the earlier held-out test probes already in the report:
  - Qwen3 improved from `0.5968` to `0.6352`
  - InternVL3 moved from `0.6340` to `0.6357`

### Status

[OK] **Completed**

---

## Session 1: COCO Gemini attribute assignment and report consolidation

**Date**: 2026-04-12
**Task**: coco-gemini-attribute-runs

### Summary

Completed the recent COCO pose Gemini-corpus attribute-assignment workflow across BLIP, Qwen3-VL-8B, and InternVL3-8B, then consolidated the finished results into `report.md` and promoted the top of the report to a 4/12 update.

### Main Changes

- Confirmed that `save/corpus_coco_pose_gemini.json` is a distinct corpus and should be passed explicitly to `utils/assign_attribute.py` to avoid falling back to `save/corpus_coco_pose.json`.
- Launched and verified the COCO pose Gemini validation runs with `utils/assign_attribute.py`:
  - Qwen3 on GPU 3 -> `save/attribute_coco_pose_gemini_qwen3_8B.json`
  - InternVL3-8B on GPU 2 -> `save/attribute_coco_pose_gemini_internvl3_8B.json`
- Diagnosed a startup failure in the `qwen` conda environment caused by `libnvJitLink` / `libcusparse` resolution, then relaunched the jobs with:
  - `LD_LIBRARY_PATH=/people/cs/o/oxx220000/.conda/envs/qwen/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH`
- Determined from `utils/assign_attribute.py` that `--dataset coco_pose` defaults to `--split val`, so the first Qwen3 / InternVL3 runs covered `val2017` only; train is not added automatically later.
- Launched and completed the follow-up train runs with `--split train --stop_at 20000`:
  - Qwen3 on GPU 0 -> `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
  - InternVL3-8B on GPU 1 -> `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`
- Updated `report.md` to record:
  - BLIP full COCO Gemini run
  - Qwen3 / InternVL3 validation results on `val2017`
  - Qwen3 / InternVL3 train results on the first `20,000` valid `train2017` images
  - a new top-level `4/12/2026 Update — COCO Gemini Attribute Assignment Results`

### Key Outputs

- Corpus definition:
  - `save/corpus_coco_pose_gemini.json`
- BLIP full run:
  - `save/attribute_coco_pose_gemini_blip.json`
- Validation outputs:
  - `save/attribute_coco_pose_gemini_qwen3_8B.json`
  - `save/attribute_coco_pose_gemini_qwen3_8B.csv`
  - `save/attribute_coco_pose_gemini_internvl3_8B.json`
  - `save/attribute_coco_pose_gemini_internvl3_8B.csv`
- Train-20k outputs:
  - `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
  - `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv`
  - `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`
  - `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`

### Runtime Notes

- Validation (`val2017`, 2,346 valid images):
  - Qwen3: 45.8 minutes
  - InternVL3-8B: 208.8 minutes
- Train subset (`train2017`, first 20,000 valid images from a 56,599-image pool):
  - Qwen3: 399.7 minutes (6.66 hours)
  - InternVL3-8B: 1709.6 minutes (28.49 hours)

### Testing

- [OK] Verified completed job endings from the corresponding logs (`Done...` and `Saved attribute annotations...`).
- [OK] Verified that validation CSV outputs contain 2,346 data rows plus header.
- [OK] Verified that train CSV outputs contain 20,000 data rows plus header.
- [OK] Verified that the top of `report.md` now starts with the 4/12 COCO Gemini update.

### Status

[OK] **Completed**

---

## Session 25: Reduce pose checkpoint disk usage

**Date**: 2026-04-15
**Task**: pose-compact-checkpoints

### Summary

Changed `train_pose_estimation.py` so its default checkpoint artifacts are compact model-only files, closer to the `alpred.py` save pattern and sufficient for the post-train val/test evaluation plus slice analysis. Full optimizer/scheduler checkpoints are now optional instead of the default.

### Why This Change Was Made

- The previous default pose-training behavior wrote:
  - `best.pth` as a full training-state checkpoint
  - `last.pth` as a full training-state checkpoint
- For Keypoint R-CNN in this repo, each full checkpoint was about `451 MB`.
- The training loop's post-train work only needs model weights:
  - reload best model
  - run held-out val evaluation
  - run held-out test evaluation
  - run attribute-driven slice analysis
- Optimizer and LR-scheduler state are not needed for that post-train analysis path.
- This makes the previous default save policy unnecessarily heavy on disk.

### Main Changes

- Added `save_model_checkpoint(...)` in `pose_estimation/engine.py`.
- Updated `train_pose_estimation.py` default save behavior:
  - `best.pth` is now model-only
  - `last.pth` is now optional rather than default
  - the compact checkpoint format includes:
    - `model_state_dict`
    - `epoch`
    - `best_ap`
- Added `--save-last-checkpoint` to `train_pose_estimation.py`:
  - default `False`
  - when enabled, writes `last.pth` as a compact model-only checkpoint
- Added `--save-resume-checkpoint` to `train_pose_estimation.py`:
  - when enabled, it writes `last_resume.pth`
  - this file contains the full training state needed for exact optimizer/scheduler continuation
- Updated the end-of-training evaluation path:
  - the training loop now reloads the compact `best.pth` / `last.pth` via `load_model_checkpoint(...)`
  - this keeps the post-train val/test evaluation and slice analysis working without full checkpoints
- Tightened resume semantics:
  - `--resume` now checks for optimizer state and errors on compact checkpoints
  - `--resume-model` remains the correct entrypoint for compact model-only checkpoints

### Validation

- [OK] `python3 -m py_compile train_pose_estimation.py pose_estimation/engine.py`
- [OK] Wrote a temporary compact checkpoint from an existing Qwen3 pose checkpoint:
  - source:
    - `log_pose/pose_keypointrcnn_train20k_qwen3_bs4_e2_rare0p97_20260412_015524/best.pth`
  - temporary output:
    - `/tmp/pose_model_only_smoke.pth`
  - observed metadata:
    - source checkpoint:
      - `has_optimizer=True`
      - `has_lr_scheduler=True`
      - `epoch=0`
      - `best_ap=0.5985`
    - compact checkpoint:
      - `has_optimizer=False`
      - `has_lr_scheduler=False`
      - `epoch=0`
      - `best_ap=0.5985`
  - observed compact size:
    - `226.1 MB`

### Disk-Usage Implication

- Previous default per run:
  - `best.pth` about `451 MB`
  - `last.pth` about `451 MB`
  - total about `902 MB`
- New intermediate compact-save design:
  - `best.pth` about `226 MB`
  - `last.pth` about `226 MB` when explicitly enabled
  - total about `452 MB` if both compact files are kept
- New actual default per run after the follow-up refinement:
  - `best.pth` about `226 MB`
  - `last.pth` not saved unless `--save-last-checkpoint true`
  - `last_resume.pth` not saved unless `--save-resume-checkpoint true`
- So compared with the old default `best+last` full-checkpoint pair, the new default reduces checkpoint disk usage by about `676 MB` per run.

### Important Runtime Note

- The two active bs8 pose-training runs launched earlier on April 15, 2026 were started before this code change.
- Those running processes keep the old save logic already loaded in memory and will not switch to the compact-save policy mid-run.

### Status

[OK] **Completed**

---

## Session 23: Launch fresh bs8 pose-training comparisons for Qwen3 and InternVL3

**Date**: 2026-04-15
**Task**: pose-train-bs8-qwen3-internvl3

### Summary

After the refreshed COCO Gemini attribute files were in place, launched two new `train_pose_estimation.py train` runs for direct comparison under the script's current defaults. One run uses the Qwen3 analysis source on GPU `3`; the other uses the InternVL3 analysis source on GPU `0`.

### Default Configuration Used

- `epochs=2`
- `batch_size=8`
- `num_workers=4`
- `lr=0.005`
- `momentum=0.9`
- `weight_decay=5e-4`
- `lr_step_size=8`
- `lr_gamma=0.1`
- `hflip_prob=0.5`
- `eval_every=1`
- `max_train_samples=20000`
- `device=cuda`
- `analyze_after_train=true`
- `analysis_rare_label_threshold=1.1`

### Important Comparison Note

- These launches are intentionally using the **current defaults**.
- Therefore they are **not** directly configuration-matched to the earlier 4/12 bs4 probe runs:
  - older Qwen3 / InternVL3 probe runs used `batch_size=4`
  - older probe naming also used `rare0p97`
- The new bs8 runs should be interpreted as fresh comparison runs under the current training defaults, not as apples-to-apples reruns of the older bs4 experiments.

### New Runs

- Qwen3 analysis-source run on GPU `3`:
  - command shape:
    - `CUDA_VISIBLE_DEVICES=3 python -u train_pose_estimation.py train --assigner qwen3 --output-dir log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_20260415_144429`
  - output dir:
    - `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_20260415_144429`
  - log:
    - `log_pose/pose_train20k_qwen3_bs8_e2_rare1p1_20260415_144429.log`
  - pid:
    - `1623280`
  - verified startup:
    - `Dataset sizes after split: train=20000, val=1173, test=1173`
    - `[Epoch 0] step 50/2500: ...`
    - GPU `3` rose above `21 GiB`

- InternVL3 analysis-source run on GPU `0`:
  - command shape:
    - `CUDA_VISIBLE_DEVICES=0 python -u train_pose_estimation.py train --assigner internvl3 --output-dir log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530`
  - output dir:
    - `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530`
  - log:
    - `log_pose/pose_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530.log`
  - pid:
    - `1624937`
  - verified startup:
    - `Dataset sizes after split: train=20000, val=1173, test=1173`
    - GPU `0` later rose to about `15.5 GiB`, confirming the process had moved onto the card even before the first step log was printed

### Status

[IN PROGRESS] **Both new bs8 pose-training runs are active**

---

## Session 24: Add `--resume-model` to pose training

**Date**: 2026-04-15
**Task**: pose-resume-model-support

### Summary

Updated the COCO pose training code so it supports an `alpred.py`-style model-only continuation mode in addition to the existing full checkpoint resume. This makes it possible to continue from a saved pose checkpoint using new optimizer / LR settings while still starting from the next epoch index recorded in the checkpoint.

### Why This Was Needed

- Before this change, `train_pose_estimation.py` only exposed `--resume`, which restores:
  - model weights
  - optimizer state
  - LR scheduler state
  - epoch index
- The user wanted an `alpred.py --resume_model`-style mode:
  - load only the saved model weights
  - keep a fresh optimizer and fresh scheduler built from the current CLI flags
  - continue training from the next epoch index, e.g. after a prior 2-epoch run, continue at epoch 3 with a new LR

### Main Changes

- In `train_pose_estimation.py`:
  - added `--resume-model`
  - enforced that `--resume` and `--resume-model` cannot be used together
  - `--resume-model` now:
    - loads model weights only
    - reads `epoch` and `best_ap` metadata when available
    - starts from `epoch + 1`
    - leaves optimizer / scheduler freshly initialized from current arguments
  - added a warning when `start_epoch >= args.epochs`, because in that case no new epochs would run unless the user increases `--epochs`
- In `pose_estimation/engine.py`:
  - added flexible model-state extraction that accepts:
    - current pose checkpoints with key `model`
    - `alpred.py`-style checkpoints with key `model_state_dict`
    - raw state-dict-like mappings
  - added `load_model_checkpoint(...)` for model-only restore
  - added `read_checkpoint_metadata(...)`
  - updated the normal `load_checkpoint(...)` path to reuse the more tolerant model-state extraction
- Fixed a latent resume bug in pose training:
  - resumed runs now carry forward `best_ap` from the checkpoint instead of resetting it to `-1.0`
  - this prevents later epochs from incorrectly overwriting a stronger previously saved `best.pth`

### Validation

- [OK] `python3 -m py_compile train_pose_estimation.py pose_estimation/engine.py`
- [OK] Model-only load smoke test in the `ad` environment using an existing pose checkpoint:
  - checkpoint:
    - `log_pose/pose_keypointrcnn_train20k_qwen3_bs4_e2_rare0p97_20260412_015524/best.pth`
  - observed output:
    - `Loaded model weights from .../best.pth (epoch 0, best AP 0.5985)`
    - returned metadata `{'epoch': 0, 'best_ap': 0.5985230961728228}`

### Usage Notes

- Full resume, keep optimizer / scheduler:
  - `python train_pose_estimation.py train --resume <ckpt> --epochs <total_epochs>`
- Model-only resume, new optimizer / scheduler:
  - `python train_pose_estimation.py train --resume-model <ckpt> --epochs <total_epochs> --lr ...`
- With the current semantics, if a checkpoint was saved at epoch index `1`, then:
  - `--resume-model ... --epochs 3`
  - will train only epoch index `2`
  - which matches the user's "train the third epoch next" example

### Status

[OK] **Completed**

---

## Session 22: Completed COCO Gemini reruns and `report.md` refresh

**Date**: 2026-04-15
**Task**: report-update-after-reruns

### Summary

Checked the April 14 follow-up COCO Gemini attribute jobs, confirmed that the remaining Qwen3 / InternVL3 runs had all finished, and updated `report.md` so it no longer claims those artifacts are still running.

### Main Findings

- The tracked follow-up job PIDs from the prior baton were no longer alive:
  - `342164` (Qwen3 `train_after20k`)
  - `383636` (InternVL3 `val`)
  - `72034` (InternVL3 canonical `train20k` rerun)
- Confirmed finished log endings:
  - Qwen3 canonical `val` rerun:
    - `Done. Processed 2346 images in 46.7 minutes`
    - `Saved attribute annotations to save/attribute_coco_pose_gemini_qwen3_8B.json`
  - InternVL3 canonical `val` rerun:
    - `Done. Processed 2346 images in 217.2 minutes`
    - `Saved attribute annotations to save/attribute_coco_pose_gemini_internvl3_8B.json`
  - InternVL3 canonical `train20k` rerun:
    - `Done. Processed 19500 images in 1876.5 minutes`
    - this was the resumed segment after an earlier `500`-row checkpoint
    - final canonical CSV still contains the requested `20,000` data rows
  - Qwen3 `train_after20k` continuation shard:
    - `Done. Processed 36599 images in 731.4 minutes`
    - `Saved attribute annotations to save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.json`
- Verified finished artifact sizes:
  - `save/attribute_coco_pose_gemini_qwen3_8B_val.csv`: `2,346` data rows
  - `save/attribute_coco_pose_gemini_internvl3_8B.csv`: `2,346` data rows
  - `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`: `20,000` data rows
  - `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.csv`: `36,599` data rows
- Reconfirmed the Qwen3 full-train coverage state:
  - the earlier `train20k` shard covers the first `20,000` valid train images
  - the new `train_after20k` shard covers the remaining `36,599`
  - together they cover all `56,599` valid `train2017` images

### Report Update

- Replaced the stale `Canonical Train20k Artifacts To Check Next` section in `report.md` with a completed-state section:
  - `Canonical Post-Cleanup Completion Update (4/15/2026)`
- The new report section records:
  - final runtimes
  - stable output paths
  - final logs
  - clarified note that the InternVL3 `31.28 h` figure is for the resumed `19,500`-image segment, not a fresh end-to-end run from zero
  - clarified note that Qwen3 full-train coverage currently lives in two explicit shards rather than one merged file

### Documentation

- Updated `.handoff.md` with a new `2026-04-15 Current Baton` completion summary.
- Updated `report.md` because the completed artifacts are now stable, human-facing results worth preserving.

### Status

[OK] **Completed**

### Next Steps

- If future agents need a compact human summary, continue using `report.md`.
- If future agents need detailed working context and decisions, continue appending to this Trellis workspace journal.
- If another cross-account or cross-tool handoff is needed, keep `.handoff.md` as the short bridge document and treat this journal as the detailed source of truth.

---

## Session 2: Two-week project context digest for future agents

**Date**: 2026-04-12
**Task**: two-week-context-digest

### Summary

Consolidated the durable project updates from recent chat sessions (roughly the last two weeks in this repo) into one agent-readable digest, so future agents can recover the important state without re-reading prior conversations.

### Main Changes

- Summarized the recent computer-vision baseline work already reflected in `report.md`:
  - COCO pose baseline with `keypointrcnn_resnet50_fpn` in `train_pose_estimation.py`
  - RTMPose-Tiny probe / evaluation results under `pose_estimation/rtmpose_assets/`
  - KITTI object-detection baseline with YOLOv8n under `object_detection/`
- Summarized the long-tail learning baseline work recorded in `.handoff.md` and `tasks/todo.md`:
  - `predict_ltl.py` gained Focal Loss, Class-Balanced Loss, Oversampling via `WeightedRandomSampler`, and LDAM variants
  - LTL runs were launched for ImageNet and CelebA and treated as active project context that should not be disrupted casually
- Summarized the paper-writing state tracked in `tasks/todo.md` and prior handoff notes:
  - Abstract, Introduction, Methodology, and Experiments were drafted
  - LTL-vs-Safefix comparisons and Active Learning equations/tables were transferred from `report.md`
  - Verification items (LaTeX compile, table alignment, equation formatting, narrative review) were marked complete
- Summarized the recent COCO Gemini attribute-assignment work:
  - `utils/blip.py` was adapted for COCO pose assignment, Gemini-corpus defaulting, `--stop_at`, and CSV export
  - BLIP completed a full COCO Gemini run at `save/attribute_coco_pose_gemini_blip.json`
  - `utils/assign_attribute.py` was used for Qwen3-VL-8B and InternVL3-8B validation and train-20k runs
- Recorded a cross-agent workflow decision:
  - `report.md` is the concise human-facing experiment record
  - `.trellis/workspace/oxu/journal-1.md` is now the preferred detailed, agent-facing context log
  - `.handoff.md` remains the short bridge for cross-account / cross-tool transfer

### Major Project Updates (Last Two Weeks)

- **COCO pose estimation baseline**
  - Dataset convention now documented clearly: `train2017` + `person_keypoints_train2017.json`, `val2017` + `person_keypoints_val2017.json`, with local test also using `val2017`
  - Effective valid-image counts are documented as `56,599` train, `2,346` val, `58,945` total
  - Keypoint R-CNN baseline in `report.md` records AP `0.489` after 2 epochs
  - RTMPose-Tiny probe in `report.md` records AP `0.697`
  - Prior investigation concluded the main pose pipeline was broadly consistent, while RTMPose config / evaluation hygiene needed careful handling

- **KITTI object detection baseline**
  - Current KITTI split is documented as deterministic filename-order partitioning: first `5000` train, next `1240` val, remaining `1241` test
  - Label space was narrowed to `Car` / `Pedestrian`
  - YOLOv8n results recorded in `report.md` show held-out test mAP@50-95 `0.563`
  - Pedestrian remains materially weaker than Car, making pedestrian-heavy slices a likely debugging target

- **Long-tail learning baselines**
  - `predict_ltl.py` work added four controlled baselines for comparison against Safefix:
    - `focal`
    - `class_weight`
    - `resample`
    - `ldam`
  - `tasks/todo.md` records these as implemented, with associated ImageNet / CelebA run commands and progress tracking
  - Important design note: when an LTL baseline is active, generated-image loading is bypassed and only original training data is used

- **NeurIPS 2026 paper drafting**
  - The paper draft work centered on the `NeurIPS 2026 Attribute/` directory
  - `tasks/todo.md` marks the main drafting and verification items complete
  - There is a filename inconsistency worth remembering:
    - older notes mention `NeurIPS 2026 Attribute/neurips_2026.tex`
    - the current directory contains `NeurIPS 2026 Attribute/main.tex`
  - Future agents should verify which TeX entrypoint is canonical before making paper edits

- **COCO Gemini attribute-assignment workflow**
  - `save/corpus_coco_pose_gemini.json` is a distinct corpus and should be passed explicitly when using `utils/assign_attribute.py`
  - `utils/assign_attribute.py --dataset coco_pose` defaults to `--split val`
  - `utils/blip.py` uses a different default split behavior (`all`), so agents must not assume the two scripts behave identically
  - Finished outputs now include:
    - BLIP full run: `save/attribute_coco_pose_gemini_blip.json`
    - Qwen3 val: `save/attribute_coco_pose_gemini_qwen3_8B.json`
    - InternVL3 val: `save/attribute_coco_pose_gemini_internvl3_8B.json`
    - Qwen3 train 20k: `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
    - InternVL3 train 20k: `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`
  - The `qwen` environment needed an explicit CUDA library path fix for Qwen3 / InternVL launches:
    - `LD_LIBRARY_PATH=/people/cs/o/oxx220000/.conda/envs/qwen/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH`

### Key Files

- `report.md` - human-facing experiment summary and metrics
- `.handoff.md` - compact cross-agent / cross-account bridge context
- `.trellis/workspace/oxu/journal-1.md` - detailed agent-facing context log
- `tasks/todo.md` - tracked LTL and paper TODO state
- `predict_ltl.py` - long-tail learning baseline implementation target
- `utils/blip.py` - BLIP COCO Gemini attribute-assignment path
- `utils/assign_attribute.py` - Qwen3 / InternVL COCO Gemini attribute-assignment path
- `train_pose_estimation.py` - Keypoint R-CNN COCO pose training entrypoint
- `object_detection/train_yolov8_kitti.py` - KITTI YOLOv8 training entrypoint
- `NeurIPS 2026 Attribute/main.tex` - current visible TeX entrypoint in the paper directory

### Testing

- [OK] Cross-checked recent durable updates against `report.md`, `.handoff.md`, `tasks/todo.md`, session history, and current output files.
- [OK] Confirmed that the recent COCO Gemini outputs and report sections referenced above exist locally.

### Status

[OK] **Completed**

### Next Steps

- Continue appending durable multi-agent context here rather than overloading `report.md`.
- Keep `report.md` focused on results and polished narrative summaries.
- Before editing the paper again, resolve the `main.tex` vs `neurips_2026.tex` entrypoint ambiguity.
- Before trusting older handoff instructions, note that `.handoff.md` references `tasks/lessons.md`, but that file is currently absent from the repo.

---

## Session 3: Pose 20k held-out slice-analysis report update

**Date**: 2026-04-12
**Task**: pose-train20k-report-update

### Summary

Verified the finished 20k COCO pose Keypoint R-CNN probe runs from logs and saved artifacts, extracted the held-out val/test metrics plus the main slice failures, and updated `report.md` so the current project summary reflects the completed pose work.

### Main Changes

- Reviewed the current pose-training diff in:
  - `train_pose_estimation.py`
  - `pose_estimation/dataset.py`
  - `pose_estimation/engine.py`
- Confirmed the new probe setup:
  - default `max_train_samples=20000`
  - deterministic split of the `2,346` valid `val2017` images into `1,173` held-out val and `1,173` held-out test images using `seed=42`
  - best-checkpoint evaluation on both held-out halves
  - post-train slice analysis via `utils/stat_pose_estimation.py`
- Verified the three completed probe runs and extracted reportable metrics:
  - Qwen3 run: held-out val AP `0.5985`, held-out test AP `0.5968`
  - InternVL3 run: held-out val AP `0.6436`, held-out test AP `0.6340`
  - BLIP run: held-out val AP `0.6427`, held-out test AP `0.6370`
- Confirmed from the slice-analysis JSON outputs that:
  - `pose=lying` is the lowest-AP recurring rare slice across all three variants
  - `pose=sitting` is the most consistent high-support hard case
  - secondary failures vary by annotation source (`time_of_day` / `weather` for Qwen3, `background_style` / `time_of_day` for InternVL3, `background_style` / `weather` / `clothes_type` for BLIP)
- Noted that the later InternVL3 artifact `log_pose/pose_keypointrcnn_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015632/` supersedes the earlier `...015544/` output, which finished at a lower held-out test AP of `0.5902`.
- Updated `report.md` with a new 4/12 subsection for the 20k probe runs and updated `.handoff.md` / `.trellis/workspace/oxu/index.md` so future agents see the current durable state first.

### Key Outputs

- Logs:
  - `log_pose/pose_train20k_qwen3_bs4_e2_rare0p97_20260412_015524.log`
  - `log_pose/pose_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015632.log`
  - `log_pose/pose_train20k_blip_bs8_e2_rare0p97_20260412_021252.log`
- Output directories:
  - `log_pose/pose_keypointrcnn_train20k_qwen3_bs4_e2_rare0p97_20260412_015524/`
  - `log_pose/pose_keypointrcnn_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015632/`
  - `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/`
- Slice summaries:
  - `log_pose/pose_keypointrcnn_train20k_qwen3_bs4_e2_rare0p97_20260412_015524/slice_analysis/manual/test_slice_stats.json`
  - `log_pose/pose_keypointrcnn_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015632/slice_analysis/manual/test_slice_stats.json`
  - `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/slice_analysis/blip/test_slice_stats.json`

### Testing

- [OK] Verified `Training complete` endings in all three logs.
- [OK] Cross-checked `history.json`, `val_metrics_best.json`, and `test_metrics.json` for each reported run.
- [OK] Cross-checked the `slice_analysis/*/test_slice_stats.json` outputs to extract the low-AP slices reported in `report.md`.
- [OK] Confirmed that the later InternVL3 artifact materially improves on the earlier duplicate run (`test AP 0.6340` vs `0.5902`).

### Status

[OK] **Completed**

### Next Steps

- Decide whether downstream pose debugging should follow the `BLIP` probe (best held-out test AP) or the `InternVL3` probe (nearly tied AP while staying closer to the VLM attribute-assignment workflow).
- If needed, compare held-out val slice summaries against held-out test slice summaries to find the most stable failure slices before any repair step.
- If this 20k probe workflow becomes the default pose baseline, consider recording the exact launch commands in a dedicated experiment ledger.

---

## Session 4: Consolidate pose artifacts under log_pose

**Date**: 2026-04-12
**Task**: pose-artifact-relocation

### Summary

Moved the existing pose-related logs and experiment artifact directories out of `logs/` and `outputs/` into a single `log_pose/` root, then updated the default pose output paths and launch docs so future pose bash commands write there directly.

---

## Session 5: SafeFix 15-minute talk slide reorder

**Date**: 2026-04-13
**Task**: safefix-talk-slide-reorder

### Summary

Moved the former appendix slide `Appendix: Full Table 1 (CelebA FR)` into the main presentation flow of the 15-minute Beamer deck so it now appears between `Experimental Setup` and `Main Results on CelebA`.

### Main Changes

- Updated `TMLR_26_SafeFix/safefix_talk_beamer_15min.tex`:
  - inserted `Full Table 1 (CelebA FR)` immediately after `Experimental Setup`
  - removed the same frame from the appendix section
- Kept the rest of the slide order unchanged.

### Testing

- [OK] Recompiled `TMLR_26_SafeFix/safefix_talk_beamer_15min.tex` with `latexmk -pdf safefix_talk_beamer_15min.tex`
- [OK] Build completed successfully and produced `TMLR_26_SafeFix/safefix_talk_beamer_15min.pdf`

### Status

[OK] **Completed**

### Notes

- `.handoff.md` and `report.md` were not updated because this change only affects presentation ordering, not active experiment state or stable result reporting.

---

## Session 6: SafeFix 15-minute talk layout cleanup

**Date**: 2026-04-13
**Task**: safefix-talk-layout-cleanup

### Summary

Adjusted the 15-minute Beamer deck layout and appendix order based on presentation feedback: removed the limitations content from the summary slide, made `Appendix: Full Table 2 (ImageNet10 FR)` the first appendix page, and changed slides 4-5 from two-column four-box layouts to vertically stacked blocks.

### Main Changes

- Updated `TMLR_26_SafeFix/safefix_talk_beamer_15min.tex`:
  - renamed `Summary and Limitations` to `Summary`
  - removed the limitations block from that slide
  - moved `Appendix: Full Table 2 (ImageNet10 FR)` to be the first appendix frame
  - reformatted `Diagnosing Rare-Case Bugs` into four top-to-bottom blocks
  - reformatted `SafeFix Design Principles` into four top-to-bottom blocks
  - tightened font size and spacing on those two stacked slides to reduce overflow

### Testing

- [OK] Recompiled `TMLR_26_SafeFix/safefix_talk_beamer_15min.tex` with `latexmk -pdf safefix_talk_beamer_15min.tex`
- [OK] Build completed successfully and produced an updated `TMLR_26_SafeFix/safefix_talk_beamer_15min.pdf`
- [NOTE] The compile log still shows small `Overfull \\vbox` warnings on the two stacked slides, but they are much smaller than the first draft after the layout change.

### Status

[OK] **Completed**

### Notes

- `.handoff.md` and `report.md` were not updated because this is presentation-only work and does not change experiment state or stable reported results.

---

## Session 7: SafeFix 15-minute talk motivation slide insertion

**Date**: 2026-04-13
**Task**: safefix-talk-motivation-slide

### Summary

Inserted a new plain motivation slide as page 2 of the 15-minute Beamer deck, based on a user-provided reference layout and using the local image `auto_driving.png`.

### Main Changes

- Updated `TMLR_26_SafeFix/safefix_talk_beamer_15min.tex`:
  - inserted a new `Motivation` slide immediately after the title page
  - used `auto_driving.png` as the main visual
  - added the motivation question text, the red autonomous-driving example text, and the Geirhos citation
  - used a `plain` frame so the slide looks closer to the provided mockup than a standard Beamer title-bar slide
  - tightened spacing and font sizes after the first compile to remove a large overflow on the inserted slide

### Testing

- [OK] Recompiled `TMLR_26_SafeFix/safefix_talk_beamer_15min.tex` with `latexmk -pdf safefix_talk_beamer_15min.tex`
- [OK] Build completed successfully and produced an updated `TMLR_26_SafeFix/safefix_talk_beamer_15min.pdf`
- [OK] The large overfull warning on the inserted motivation slide was eliminated after spacing cleanup

### Status

[OK] **Completed**

### Notes

- `.handoff.md` and `report.md` were not updated because this is presentation-only work and does not affect experiment state or stable result reporting.

---

## Session 8: SafeFix motivation slide style normalization

**Date**: 2026-04-13
**Task**: safefix-talk-motivation-style

### Summary

Changed the newly inserted motivation slide from a mockup-style plain page back to the deck's normal Beamer frame styling so it matches the rest of the presentation background and title treatment.

### Main Changes

- Updated `TMLR_26_SafeFix/safefix_talk_beamer_15min.tex`:
  - replaced the inserted `plain` motivation frame with a standard `\begin{frame}[t]{Motivation}`
  - kept the same question text, `auto_driving.png`, red example text, and citation
  - preserved the overall layout while restoring the standard slide title bar and background

### Testing

- [OK] Recompiled `TMLR_26_SafeFix/safefix_talk_beamer_15min.tex` with `latexmk -pdf safefix_talk_beamer_15min.tex`
- [OK] Build completed successfully and produced an updated `TMLR_26_SafeFix/safefix_talk_beamer_15min.pdf`

### Status

[OK] **Completed**

### Notes

- `.handoff.md` and `report.md` were not updated because this is presentation-only work and does not affect experiment state or stable result reporting.

### Follow-up Edit

- Reduced the red example text on the motivation slide from `\fontsize{18}{21}` to `\fontsize{13.5}{15.75}`, matching the requested `0.75x` scale.
- Recompiled `TMLR_26_SafeFix/safefix_talk_beamer_15min.tex` successfully after the size adjustment.

### Main Changes

- Created `log_pose/` as the canonical root for pose experiment artifacts.
- Moved pose-related runtime logs and pid files from `logs/` into `log_pose/`, including:
  - `pose_*`
  - `rtmpose_*`
  - `assign_attribute_coco_pose*`
  - `blip_coco_pose*`
- Moved pose-related artifact directories from `outputs/` into `log_pose/`, including:
  - `pose_keypointrcnn_*`
  - `rtmpose_tiny_coco_gpu3_*`
- Updated future defaults and launch examples:
  - `train_pose_estimation.py` now defaults to `log_pose/pose_keypointrcnn`, `log_pose/pose_eval_results.json`, `log_pose/pose_test_results.json`, and `log_pose/pose_infer.jpg`
  - `utils/stat_pose_estimation.py` now defaults slice-stat outputs under `log_pose/pose_slice_stats/`
  - `scripts/run_pose_coco_nohup.sh` now writes both logs and artifact directories into `log_pose/`
  - pose launch snippets in `draft.sh` now target `log_pose/`
  - `pose_estimation/README.md`, `report.md`, and `.handoff.md` now reference `log_pose/`
- Added `log_pose/` to `.gitignore` to match the existing treatment of large generated experiment artifacts.

### Key Outputs

- Canonical pose artifact root:
  - `log_pose/`
- Example relocated logs:
  - `log_pose/pose_train20k_qwen3_bs4_e2_rare0p97_20260412_015524.log`
  - `log_pose/pose_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015632.log`
  - `log_pose/pose_coco_pretrained_gpu3_ad_e2.log`
  - `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_gpu0_20260410_193459.log`
- Example relocated artifact directories:
  - `log_pose/pose_keypointrcnn_ad_gpu3_e2/`
  - `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/`
  - `log_pose/pose_keypointrcnn_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015632/`
  - `log_pose/rtmpose_tiny_coco_gpu3_probe/`

### Testing

- [OK] Verified that the pose-related files are present under `log_pose/`.
- [OK] Verified that the corresponding pose-related files are no longer present at the top level of `logs/` and `outputs/`.
- [OK] Verified that the default output paths in the pose code and launch scripts now point to `log_pose/`.

### Status

[OK] **Completed**

### Next Steps

- Use `log_pose/` consistently for future pose runs and avoid sending pose bash outputs to `outputs/`.
- If desired later, normalize the internal path strings stored inside old historical logs / metric JSON files; the physical files have already been moved, but historical text payloads still mention their original write locations.

---

## Session 5: Gemini pose corpus label cleanup

**Date**: 2026-04-12
**Task**: pose-gemini-corpus-cleanup

### Summary

Updated the Gemini COCO pose corpus label set so future attribute-assignment runs no longer emit `weather=unknown`, while adding a few conservative extra labels in other fields where the previous vocabulary looked unnecessarily narrow.

### Main Changes

- Edited `save/corpus_coco_pose_gemini.json`.
- Removed `unknown` from the `weather` label set, leaving:
  - `sunny`
  - `cloudy`
  - `rainy`
  - `snowy`
  - `clear`
- Added a small number of extra labels outside `weather`:
  - `clothes_color`: added `brown`, `pink`
  - `clothes_type`: added `uniform`
  - `pose`: added `walking`
- Intentionally left `time_of_day=unknown` unchanged, because that ambiguity is still plausible for many COCO images and forcing an incorrect time bucket would likely add noise.

### Key Files

- `save/corpus_coco_pose_gemini.json`

### Testing

- [OK] Verified the edited corpus file remains valid JSON.
- [OK] Confirmed the `weather` label list no longer includes `unknown`.

### Status

[OK] **Completed**

### Next Steps

- Future Gemini-corpus attribute runs will use the revised label space automatically.
- Existing saved attribute CSV/JSON outputs are unchanged; they still contain historical `weather=unknown` values from earlier runs.

---

## Session 6: Relaunch Gemini pose attribute refresh jobs

**Date**: 2026-04-12
**Task**: relaunch-pose-gemini-assigners

### Summary

After removing `weather=unknown` from the Gemini pose corpus, launched fresh Qwen3 and InternVL3 COCO pose attribute-assignment jobs so the canonical `save/attribute_coco_pose_gemini_*` files will be regenerated with the updated label space.

### Main Changes

- Confirmed that `train_pose_estimation.py` already points to the canonical files:
  - Qwen3 val: `save/attribute_coco_pose_gemini_qwen3_8B.json`
  - Qwen3 train20k: `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
  - InternVL3 val: `save/attribute_coco_pose_gemini_internvl3_8B.json`
  - InternVL3 train20k: `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`
- Therefore no code-path change was needed for future pose training: once the new jobs finish, later `train_pose_estimation.py train --assigner qwen3|internvl3` runs will automatically consume the refreshed files.
- Launched persistent `nohup` jobs outside the sandbox:
  - GPU `0`: Qwen3 refresh (`val` then `train --stop_at 20000`)
  - GPU `3`: InternVL3 refresh (`val` then `train --stop_at 20000`)
- Both launches use `--max_memory_per_gpu 23.9`.

### Key Outputs

- Qwen3 refresh:
  - PID: `793765`
  - Log: `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_refresh_gpu0_20260412_160952.log`
- InternVL3 refresh:
  - PID: `793766`
  - Log: `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_refresh_gpu3_20260412_160952.log`

### Testing

- [OK] Verified both PIDs exist as live background shells.
- [OK] Verified both logs contain startup output and model-loading messages.
- [OK] Verified GPUs `0` and `3` became active immediately after launch.

### Status

[OK] **Running**

### Next Steps

- Monitor the two refresh logs until both finish `val`, then continue into `train20k`.
- After completion, any future pose training that uses the canonical Qwen3 / InternVL3 attribute files will automatically see the updated corpus outputs.

---

## Session 7: Make pose slice analysis tolerate missing eval attributes

**Date**: 2026-04-12
**Task**: pose-analysis-na-fallback

### Summary

Adjusted the pose post-train analysis path so missing eval-side attribute assignments no longer show up as a failed slice-analysis step. The analysis now writes an explicit `NA` summary instead, and `assign-after-train` now defaults to running only the `train` split.

### Main Changes

- Updated `train_pose_estimation.py`:
  - changed `--assigner-splits` default from `val,train` to `train`
  - this avoids spending extra time recomputing eval-side assignments during `--assign-after-train` unless requested explicitly
- Updated `utils/stat_pose_estimation.py`:
  - if `--attribute-file` is missing, write a JSON summary with `status: "NA"` and `reason: "Missing eval attribute file: ..."` plus an empty-header CSV, then return without raising
  - if `--train-attribute-file` is missing, continue analysis and leave train-side share fields unavailable instead of failing
  - CSV export now writes `NA` for `None` fields instead of blank cells

### Key Files

- `train_pose_estimation.py`
- `utils/stat_pose_estimation.py`

### Testing

- [OK] `python3 -m py_compile train_pose_estimation.py utils/stat_pose_estimation.py`
- [OK] Smoke-tested the missing-file path in the `ad` environment with:
  - `python utils/stat_pose_estimation.py --predictions-json log_pose/pose_keypointrcnn_train20k_qwen3_bs4_e2_rare0p97_20260412_015524/test_predictions.json --attribute-file save/__missing_pose_attrs__.json --attribute-split val_full --train-attribute-file save/attribute_coco_pose_gemini_train20k_qwen3_8B.json --train-attribute-split train --eval-split test --output-json /tmp/pose_missing_attr_smoke.json --output-csv /tmp/pose_missing_attr_smoke.csv`
- [OK] Verified the smoke output JSON contains:
  - `status: "NA"`
  - `reason: "Missing eval attribute file: save/__missing_pose_attrs__.json"`

### Status

[OK] **Completed**

### Next Steps

- If future agents want held-out val/test slice analysis, they still need an eval-side attribute file that covers `val2017`.
- If eval-side attributes are intentionally skipped, the training pipeline will now finish cleanly and leave `NA` placeholders under `slice_analysis/` instead of printing a failure.

---

## Session 8: Replace val-first pose assign refresh with train-only jobs

**Date**: 2026-04-12
**Task**: pose-train-only-assign-relaunch

### Summary

Stopped the earlier Qwen3 / InternVL3 Gemini refresh jobs because they were still running `val` first, then relaunched the canonical attribute-assignment jobs as `train`-only runs on GPU `0` and GPU `3`.

### Main Changes

- Confirmed the earlier background shells were still executing:
  - Qwen3: `--split val` first, then `train`
  - InternVL3: `--split val` first, then `train`
- Stopped the old refresh jobs:
  - shell PIDs `793765` and `793766`
  - remaining InternVL3 val child PID `793812`
- Relaunched `train --stop_at 20000` only:
  - Qwen3 on GPU `0`
  - InternVL3 on GPU `3`
- Verified the live Python commands now both use:
  - `--split train --stop_at 20000`
  - `save/corpus_coco_pose_gemini.json`
  - canonical output files under `save/attribute_coco_pose_gemini_train20k_*`

### Key Outputs

- Active Qwen3 process:
  - PID: `801242`
  - Command: `utils/assign_attribute.py --dataset coco_pose --split train --stop_at 20000 --model Qwen/Qwen3-VL-8B-Instruct ...`
  - Active log: `log_pose/NA/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_only_gpu0_20260412_162135.log`
- Active InternVL3 process:
  - PID: `802309`
  - Command: `utils/assign_attribute.py --dataset coco_pose --split train --stop_at 20000 --model OpenGVLab/InternVL3-8B ...`
  - Active log: `log_pose/NA/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_only_gpu3_20260412_162238.log`
- One failed duplicate Qwen3 relaunch should be ignored:
  - `log_pose/NA/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_only_gpu0_20260412_162238.log`
  - It failed with CUDA OOM because GPU `0` was already occupied by the live Qwen3 train-only process.

### Testing

- [OK] Verified old shell PIDs `793765` / `793766` no longer exist.
- [OK] Verified remaining val child PID `793812` was terminated.
- [OK] Verified live processes with `ps -fp 801242 802309`.
- [OK] Verified both GPUs show train-only Python processes via `nvidia-smi --query-compute-apps=...`.

### Status

[OK] **Running**

### Next Steps

- Monitor:
  - `log_pose/NA/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_only_gpu0_20260412_162135.log`
  - `log_pose/NA/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_only_gpu3_20260412_162238.log`
- Ignore the failed duplicate Qwen3 log at `...162238.log`; the real running Qwen3 process is PID `801242`.

---

## Session 9: Add canonical train20k artifact pointers to report and launch BLIP train20k

**Date**: 2026-04-12
**Task**: report-train20k-artifacts-and-blip-launch

### Summary

Updated `report.md` so future report edits can immediately find the exact post-cleanup train20k logs and outputs to inspect, then launched the missing BLIP Gemini-corpus `train20k` attribute-assignment job on GPU `2`.

### Main Changes

- Updated `report.md` in the COCO Gemini train-run section:
  - kept the historical completed Qwen3 / InternVL3 `20260410` train20k runs
  - added a new `Canonical Train20k Artifacts To Check Next` subsection
  - recorded the exact canonical post-cleanup log / JSON / CSV paths for:
    - Qwen3 refresh on GPU `0`
    - InternVL3 refresh on GPU `3`
    - BLIP train20k on GPU `2`
  - added explicit completion-line guidance so later agents know what to grep for in each log
- Launched BLIP train20k with the Gemini corpus on GPU `2`:
  - `utils/blip.py --dataset coco_pose --split train --stop_at 20000 --with_attribute_selection --corpus save/corpus_coco_pose_gemini.json`
  - outputs:
    - `save/attribute_coco_pose_gemini_train20k_blip.json`
    - `save/attribute_coco_pose_gemini_train20k_blip.csv`
    - `save/labels_coco_pose_gemini_train20k_blip.npy`
    - `save/label_coco_pose_gemini_train20k_blip.npy`
    - `save/train_idx_coco_pose_gemini_train20k_blip.npy`
    - `save/val_idx_coco_pose_gemini_train20k_blip.npy`

### Key Outputs

- BLIP train20k launch:
  - PID: `804337`
  - Log: `log_pose/blip_coco_pose_gemini_train20k_gpu2_20260412_162538.log`
- Active canonical post-cleanup train20k logs:
  - Qwen3: `log_pose/NA/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_only_gpu0_20260412_162135.log`
  - InternVL3: `log_pose/NA/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_only_gpu3_20260412_162238.log`
  - BLIP: `log_pose/blip_coco_pose_gemini_train20k_gpu2_20260412_162538.log`

### Testing

- [OK] Verified `report.md` now contains the canonical artifact checklist section.
- [OK] Verified live BLIP command with `ps -fp 804337`.
- [OK] Verified BLIP log startup shows:
  - `Loaded COCO pose split 'train' ... 56599 valid images`
  - `Will process only the first 20000 images (--stop_at 20000)`
  - `Match clothes_color`
- [OK] Verified the BLIP side-output files were initialized:
  - labels/text labels/train idx/val idx

### Status

[OK] **Running**

### Next Steps

- Monitor the three canonical post-cleanup train20k logs already recorded in `report.md`.
- Once the BLIP train20k run finishes, update `report.md` with its runtime and final completion lines so the three-model train20k summary is fully symmetric.

---

## Session 10: Verify BLIP train20k completion

**Date**: 2026-04-12
**Task**: verify-blip-train20k-finish

### Summary

Confirmed that the BLIP Gemini-corpus `train20k` COCO pose attribute-assignment run on GPU `2` finished successfully, then updated the report and handoff notes so the recorded status is no longer stale.

### Main Changes

- Verified that PID `804337` no longer exists.
- Verified the BLIP log ends with:
  - `Saved attributes to save/attribute_coco_pose_gemini_train20k_blip.json`
  - `Saved attribute CSV to save/attribute_coco_pose_gemini_train20k_blip.csv`
- Verified output timestamps around completion time:
  - completed at approximately `2026-04-12 16:38 CDT`
- Updated `report.md`:
  - changed the BLIP canonical train20k status from running to completed
  - added explicit BLIP completion lines
  - changed the follow-up note from in-progress to completed

### Key Outputs

- BLIP log:
  - `log_pose/blip_coco_pose_gemini_train20k_gpu2_20260412_162538.log`
- BLIP outputs:
  - `save/attribute_coco_pose_gemini_train20k_blip.json`
  - `save/attribute_coco_pose_gemini_train20k_blip.csv`
  - `save/labels_coco_pose_gemini_train20k_blip.npy`
  - `save/label_coco_pose_gemini_train20k_blip.npy`
  - `save/train_idx_coco_pose_gemini_train20k_blip.npy`
  - `save/val_idx_coco_pose_gemini_train20k_blip.npy`

### Testing

- [OK] `ps -fp 804337` returns no live process.
- [OK] `tail -n 40 log_pose/blip_coco_pose_gemini_train20k_gpu2_20260412_162538.log` shows successful final save lines.
- [OK] `ls -lh` confirms the final JSON and CSV exist.

### Status

[OK] **Completed**

### Next Steps

- Wait for the Qwen3 and InternVL3 post-cleanup train20k refresh jobs to finish.
- Once those two complete, update `report.md` with their final completion lines as well so the canonical post-cleanup three-model train20k block is fully finished.

---

## Session 11: Run BLIP-based test slice analysis for the bs8 pose probe

**Date**: 2026-04-12
**Task**: blip-postcleanup-bs8-test-analysis

### Summary

Used the only available `bs=8` pose probe result (the BLIP-tagged `train20k` run) and reran held-out `test` slice analysis with the newly finished BLIP `train20k` attribute file for train-side distribution accounting.

### Main Changes

- Clarified the experiment-selection rule for this request:
  - among current pose probe artifacts, only one `bs=8` run exists
  - therefore the "worst `bs=8`" run is, by default, `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/`
- Ran:
  - predictions: `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/test_predictions.json`
  - eval attributes: `save/attribute_coco_pose_gemini_blip.json` with `attribute_split=all`
  - train attributes: `save/attribute_coco_pose_gemini_train20k_blip.json` with `train_attribute_split=train`
  - corpus: `save/corpus_coco_pose_gemini.json`
- Saved the refreshed test slice analysis under a separate path so the earlier BLIP slice-analysis outputs were not overwritten.

### Key Outputs

- Refreshed test slice analysis:
  - `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/slice_analysis/blip_postcleanup_train20k/test_slice_stats.json`
  - `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/slice_analysis/blip_postcleanup_train20k/test_slice_stats.csv`

### Main Findings

- Overall metrics are unchanged because the predictions are the same:
  - held-out test AP `0.6370`
  - held-out test AR `0.7009`
- Top failure slices in this refreshed BLIP-based analysis:
  - `pose=lying`: `AP=0.4195`, `delta=-0.2175`, `type=rare_case`
  - `pose=sitting`: `AP=0.5633`, `delta=-0.0737`, `type=hard_case`
  - `background_style=indoors`: `AP=0.5977`, `delta=-0.0393`, `type=hard_case`
  - `weather=unknown`: `AP=0.6041`, `delta=-0.0329`, `type=distribution_shift`
- Important caveat:
  - `weather=unknown` still appears because the eval-side BLIP file `save/attribute_coco_pose_gemini_blip.json` was generated before the corpus cleanup
  - train-side shares now come from the new post-cleanup `save/attribute_coco_pose_gemini_train20k_blip.json`
  - therefore this output is a mixed-corpus analysis: refreshed on the train side, historical on the eval side

### Testing

- [OK] Ran `utils/stat_pose_estimation.py` in the `ad` environment with the paths above.
- [OK] Verified the JSON and CSV outputs were written successfully.
- [OK] Verified the printed summary matches the expected held-out test AP for the selected `bs=8` run.

### Status

[OK] **Completed**

### Next Steps

- If a fully post-cleanup BLIP analysis is required, rerun BLIP attribute assignment on the eval side (`val` or `all`) and then rerun `utils/stat_pose_estimation.py`.
- Until then, use the refreshed output above as the best available BLIP-based test slice analysis that incorporates the new BLIP `train20k` file.

---

## Session 12: Diagnose stale BLIP eval file and relaunch val on GPU 2

**Date**: 2026-04-12
**Task**: fix-blip-eval-unknown

### Summary

Diagnosed why `weather=unknown` still appeared in BLIP-based slice analysis after the corpus cleanup, confirmed the issue was a stale eval-side BLIP attribute file, then relaunched BLIP on the COCO pose `val` split on GPU `2` to regenerate a clean eval-side file.

### Main Changes

- Verified the corpus itself is correct:
  - `save/corpus_coco_pose_gemini.json` now has `weather = ['sunny', 'cloudy', 'rainy', 'snowy', 'clear']`
- Compared BLIP attribute files:
  - `save/attribute_coco_pose_gemini_blip.json` still contains `unknown` in `weather` (`16049` rows)
  - `save/attribute_coco_pose_gemini_train20k_blip.json` contains `0` `unknown` rows in `weather`
- Root cause:
  - the old full BLIP file was created before the corpus cleanup
  - the new train20k BLIP file was created after the cleanup
  - mixed use of these two files caused `weather=unknown` to remain in eval-side slice analysis
- Launched BLIP `val` rerun on GPU `2`:
  - output JSON: `save/attribute_coco_pose_gemini_blip_val.json`
  - output CSV: `save/attribute_coco_pose_gemini_blip_val.csv`
  - log: `log_pose/blip_coco_pose_gemini_val_gpu2_20260412_171035.log`
- Updated `train_pose_estimation.py` BLIP defaults so future BLIP analysis uses:
  - eval: `save/attribute_coco_pose_gemini_blip_val.json` with `attribute_split=val_full`
  - train: `save/attribute_coco_pose_gemini_train20k_blip.json` with `train_attribute_split=train`

### Key Outputs

- Running BLIP val process:
  - PID: `826840`
  - Log: `log_pose/blip_coco_pose_gemini_val_gpu2_20260412_171035.log`
- New eval-side targets:
  - `save/attribute_coco_pose_gemini_blip_val.json`
  - `save/attribute_coco_pose_gemini_blip_val.csv`
  - `save/labels_coco_pose_gemini_val_blip.npy`
  - `save/label_coco_pose_gemini_val_blip.npy`
  - `save/train_idx_coco_pose_gemini_val_blip.npy`
  - `save/val_idx_coco_pose_gemini_val_blip.npy`

### Testing

- [OK] Confirmed `weather_words` in the corpus no longer include `unknown`.
- [OK] Counted `unknown` values:
  - old BLIP full file: `16049`
  - new BLIP train20k file: `0`
- [OK] `python3 -m py_compile train_pose_estimation.py`
- [OK] Verified the new BLIP val job is running with `ps -fp 826840`.
- [OK] Verified the new BLIP val log shows:
  - `Loaded COCO pose split 'val' ... 2346 valid images`
  - `Match clothes_color`

### Status

[OK] **Running**

### Next Steps

- Wait for `log_pose/blip_coco_pose_gemini_val_gpu2_20260412_171035.log` to finish.
- Then rerun BLIP-based `test` slice analysis using:
  - eval file `save/attribute_coco_pose_gemini_blip_val.json`
  - train file `save/attribute_coco_pose_gemini_train20k_blip.json`

---

## Session 13: Verify BLIP val completion and rerun clean BLIP test slice analysis

**Date**: 2026-04-12
**Task**: blip-clean-test-analysis

### Summary

Confirmed that the BLIP `val` rerun finished successfully with no `weather=unknown` labels, then reran held-out `test` slice analysis for the `bs=8` pose probe using the new BLIP eval file plus the new BLIP train20k file.

### Main Changes

- Verified BLIP `val` completion:
  - `ps -fp 826840` returned no live process
  - log ended with:
    - `Saved attributes to save/attribute_coco_pose_gemini_blip_val.json`
    - `Saved attribute CSV to save/attribute_coco_pose_gemini_blip_val.csv`
- Verified the new eval-side BLIP file:
  - `save/attribute_coco_pose_gemini_blip_val.json`
  - contains `0` `weather=unknown` labels
  - weather values are exactly `sunny`, `clear`, `cloudy`, `rainy`, `snowy`
- Reran clean BLIP held-out `test` slice analysis with:
  - predictions: `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/test_predictions.json`
  - eval file: `save/attribute_coco_pose_gemini_blip_val.json`
  - train file: `save/attribute_coco_pose_gemini_train20k_blip.json`
  - corpus: `save/corpus_coco_pose_gemini.json`

### Key Outputs

- Clean BLIP eval file:
  - `save/attribute_coco_pose_gemini_blip_val.json`
  - `save/attribute_coco_pose_gemini_blip_val.csv`
- Clean BLIP held-out test slice analysis:
  - `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/slice_analysis/blip_postcleanup_clean/test_slice_stats.json`
  - `log_pose/pose_keypointrcnn_train20k_blip_bs8_e2_rare0p97_20260412_021252/slice_analysis/blip_postcleanup_clean/test_slice_stats.csv`

### Main Findings

- Overall held-out test AP remains `0.6370` because the predictions are unchanged.
- `weather=unknown` is gone in the clean BLIP analysis.
- The main failure slices in the clean BLIP analysis are now:
  - `pose=lying`: `AP=0.4112`, `delta=-0.2258`, `type=rare_case`
  - `clothes_color=brown`: `AP=0.5458`, `delta=-0.0912`, `type=rare_case`
  - `pose=sitting`: `AP=0.5654`, `delta=-0.0716`, `type=hard_case`
  - `background_style=indoors`: `AP=0.5977`, `delta=-0.0393`, `type=hard_case`
- Weather-side failures are now:
  - `rainy`
  - `snowy`
  - `sunny`
  - with no `unknown`

### Testing

- [OK] Verified the new BLIP eval JSON has `unknown_count = 0` for `weather`.
- [OK] Verified the clean slice-analysis JSON has no `weather=unknown` entry.
- [OK] Updated `report.md` to record the clean BLIP val rerun and the clean test slice-analysis output path.

### Status

[OK] **Completed**

### Next Steps

- If needed, rerun BLIP held-out `val` slice analysis in the same clean configuration for symmetry with the clean test output.
- Wait for the Qwen3 / InternVL3 train20k refresh jobs to finish if a clean three-model comparison is still desired.

---

## Session 14: Create bxz handoff folder for clean BLIP pose JSONs

**Date**: 2026-04-12
**Task**: bxz-json-handoff

### Summary

Created a dedicated `bxz/` folder containing copied JSON artifacts for the clean BLIP-based pose slice analysis, plus a concise `README.md` so a collaborator can consume the package without tracing the full repo history.

### Main Changes

- Created `bxz/`.
- Copied the main clean BLIP pose-analysis JSON files into `bxz/`:
  - `corpus_coco_pose_gemini.json`
  - `attribute_coco_pose_gemini_blip_val.json`
  - `attribute_coco_pose_gemini_train20k_blip.json`
  - `test_predictions.json`
  - `test_metrics.json`
  - `test_slice_stats.json`
- Added `bxz/README.md` with:
  - what each file is
  - which file to read first
  - the main rare slices
  - the original canonical source paths
  - the note that the old full BLIP file was intentionally excluded because it still contained `weather=unknown`

### Key Outputs

- `bxz/README.md`
- `bxz/corpus_coco_pose_gemini.json`
- `bxz/attribute_coco_pose_gemini_blip_val.json`
- `bxz/attribute_coco_pose_gemini_train20k_blip.json`
- `bxz/test_predictions.json`
- `bxz/test_metrics.json`
- `bxz/test_slice_stats.json`

### Testing

- [OK] Verified the copied files exist under `bxz/`.
- [OK] Verified `bxz/README.md` documents the package and source paths correctly.

### Status

[OK] **Completed**

### Next Steps

- If `bxz` also needs CSV or log files later, add them explicitly rather than moving the canonical originals.

---

## Session 5: Create KITTI object-attribute corpus task context

**Date**: 2026-04-12
**Task**: kitti-object-attribute-corpus

### Summary

Created and activated a new Trellis task dedicated to designing a KITTI object-detection attribute corpus for the first `5000` YOLO-format images, separate from the earlier COCO BLIP assignment task.

### Main Changes

- Read the current Trellis workflow and project handoff context before switching tasks.
- Confirmed the prior current task was `.trellis/tasks/04-10-blip-coco-assign` and should remain separate.
- Created a new task directory:
  - `.trellis/tasks/04-12-kitti-object-attribute-corpus/`
- Wrote `prd.md` for the new task with these core requirements:
  - inspect `save/corpus_coco_pose_gemini.json`
  - inspect `utils/assign_attribute.py`
  - inspect `/data/oxx220000/kitti_yolov8_first5000`
  - design an editable KITTI object corpus aligned with the spirit of Appendix Table 5 in `arXiv:2501.16751`
  - save the output as a GPT-style corpus such as `save/corpus_kitti_object_gpt.json`
- Initialized Trellis context files for the task as `backend`:
  - `implement.jsonl`
  - `check.jsonl`
  - `debug.jsonl`
- Activated the new task as the current task with `task.py start`.

### Key Files

- `.trellis/tasks/04-12-kitti-object-attribute-corpus/task.json`
- `.trellis/tasks/04-12-kitti-object-attribute-corpus/prd.md`
- `.trellis/tasks/04-12-kitti-object-attribute-corpus/implement.jsonl`
- `.trellis/tasks/04-12-kitti-object-attribute-corpus/check.jsonl`
- `.trellis/tasks/04-12-kitti-object-attribute-corpus/debug.jsonl`

### Testing

- [OK] Verified `python3 ./.trellis/scripts/task.py create ...` created the new task directory.
- [OK] Verified `task.py init-context ... backend` created the context jsonl files.
- [OK] Verified `task.py start ...` switched `.current-task` to `04-12-kitti-object-attribute-corpus`.
- [OK] Verified `get_context.py` now reports `kitti-object-attribute-corpus` as the current task.

### Status

[OK] **Completed**

### Next Steps

- Read the new task `prd.md` as the source of truth for the KITTI corpus work.
- Inspect the existing COCO corpus format, `assign_attribute.py`, and the KITTI first-5000 dataset before proposing the final attribute set.

---

## Session 6: KITTI object corpus for first-5000 driving images

**Date**: 2026-04-12
**Task**: kitti-object-attribute-corpus

### Summary

Designed and added a new KITTI object-detection corpus for the first `5000` training images, aimed at controllable image editing and downstream slice analysis, then updated `utils/assign_attribute.py` so the prompts and KITTI split handling fit that workflow better.

### Main Changes

- Inspected the existing corpus structure and confirmed `utils/assign_attribute.py` only requires a top-level `{"Corpus": ...}` object with per-attribute `data`, `word`, `prompt`, `type`, and `question` fields.
- Inspected the KITTI YOLO root at `/data/oxx220000/kitti_yolov8_first5000` and verified:
  - `images/train` contains the first `5000` images
  - `labels/train` contains non-empty YOLO labels for all `5000` images
  - the current yaml lists 8 classes (`Car`, `Van`, `Truck`, `Pedestrian`, `Person_sitting`, `Cyclist`, `Tram`, `Misc`)
- Measured train-label counts to ground the corpus design in the actual first-5000 data:
  - `Car=19075`
  - `Pedestrian=3042`
  - `Cyclist=1124`
  - `Van=1957`
  - `Truck=724`
  - `Tram=344`
  - `Person_sitting=145`
  - `Misc=651`
- Measured object-scale distribution and found the set is dominated by small and medium boxes, which supported keeping an editable `object_scale` attribute.
- Created `save/corpus_kitti_object_gpt.json` with 10 attributes chosen to stay visually inferable and editable for ControlNet / generative repair workflows:
  - `color`
  - `object_scale`
  - `viewpoint`
  - `occlusion`
  - `truncation`
  - `motion_state`
  - `background_style`
  - `traffic_density`
  - `lighting_condition`
  - `weather`
- Updated `utils/assign_attribute.py` to better support this corpus:
  - changed prompt wording from `single word` to `chosen option` so multi-word labels like `city street` remain valid
  - generalized `background_style`, `occlusion`, and `viewpoint` prompts beyond pose-only wording
  - added explicit prompt templates for `object_scale`, `truncation`, and `motion_state`
  - improved the text-only remap prompt to refer to `chosen option`
- Cross-checked the paper intent directly from the arXiv HTML after a background lookup timed out:
  - HiBug2 explicitly frames attribute generation around `main object`, `background`, and `global` categories
  - for object detection, it emphasizes contextual scene factors plus difficulty-related properties such as occlusion and small objects
  - this confirmed the new KITTI corpus should mix target attributes (`color`, `object_scale`, `viewpoint`, `occlusion`, `truncation`, `motion_state`) with scene/global attributes (`background_style`, `traffic_density`, `lighting_condition`, `weather`)
- Fixed KITTI split handling in `utils/assign_attribute.py`:
  - `collect_kitti_dataset()` now respects `train`, `val`, `test`, and `all`
  - when `--dataset kitti` is used without an explicit `--split`, the script now defaults to `train` so the first-5000 training subset is the default path

### Key Files

- `save/corpus_kitti_object_gpt.json`
- `utils/assign_attribute.py`
- `/data/oxx220000/kitti_yolov8_first5000/kitti_first5000.yaml`

### Suggested Usage

- To annotate the first `5000` KITTI training images with the new corpus:
  - `python utils/assign_attribute.py --dataset kitti --kitti_root /data/oxx220000/kitti_yolov8_first5000 --split train --corpus save/corpus_kitti_object_gpt.json`

### Testing

- [OK] `python3 -m py_compile utils/assign_attribute.py`
- [OK] Loaded `save/corpus_kitti_object_gpt.json` with Python and verified all 10 attributes contain the required schema keys.
- [OK] Confirmed no `single word` prompt text remains in `utils/assign_attribute.py`.

### Status

[OK] **Completed**

### Next Steps

- If needed, run a small KITTI smoke assignment with the new corpus to inspect label quality before a long full-train annotation run.
- If later experiments must stay strictly aligned with a narrower `Car` / `Pedestrian` label space, consider a second corpus variant specialized for that two-class baseline.

---

## Session 15: Verify incomplete COCO train20k reruns and restart Qwen3 / InternVL3

**Date**: 2026-04-13
**Task**: restart-coco-train20k-qwen3-internvl3

### Summary

Checked the refreshed Gemini-corpus COCO pose `train20k` Qwen3 / InternVL3 outputs after the user said they had been rerun, found that both canonical CSVs were incomplete with no live processes attached, then relaunched fresh clean jobs on GPU `0` and GPU `1`.

### Main Changes

- Verified the stale state before restart:
  - the old 4/12 train-only PIDs `801242` and `802309` no longer existed
  - `GPU 0` and `GPU 1` were idle
  - `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv` had `18,000` data rows (`18,001` lines including header)
  - `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv` had `4,000` data rows (`4,001` lines including header)
  - the stale partial checkpoints had newer mtimes than the last completed JSONs:
    - Qwen3 partial: `save/attribute_coco_pose_gemini_train20k_qwen3_8B_partial.json` (`2026-04-12 22:19 CDT`)
    - InternVL3 partial: `save/attribute_coco_pose_gemini_train20k_internvl3_8B_partial.json` (`2026-04-12 22:13 CDT`)
- Confirmed from `utils/assign_attribute.py` that a relaunch without `--resume` performs a clean restart and rewrites the canonical CSV / partial checkpoint paths, so no code change was needed.
- Relaunched the canonical Gemini-corpus `train20k` jobs with the existing repo workflow under the `qwen` conda env plus the `LD_LIBRARY_PATH` nvjitlink fix:
  - Qwen3 on GPU `0`
  - InternVL3 on GPU `1`
- Verified immediately after relaunch that both jobs:
  - loaded their models successfully
  - loaded the COCO pose `train` split (`56,599` valid images)
  - applied `--stop_at 20000`
  - reached the first progress line `[1/20000]`
- Verified with `ps` and `nvidia-smi` that:
  - `PID 41453` is the live Qwen3 restart on GPU `0`
  - `PID 41464` is the live InternVL3 restart on GPU `1`
  - GPUs `2` and `3` are occupied by unrelated processes from another user, so using `0/1` matched the requested restart plan without interference

### Key Logs

- Qwen3 restart:
  - `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_restart_gpu0_20260413_131326.log`
  - `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_restart_gpu0_20260413_131326.pid`
- InternVL3 restart:
  - `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_restart_gpu1_20260413_131327.log`
  - `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_restart_gpu1_20260413_131327.pid`

### Canonical Output Paths Being Regenerated

- `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
- `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv`
- `save/attribute_coco_pose_gemini_train20k_qwen3_8B_partial.json`
- `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`
- `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`
- `save/attribute_coco_pose_gemini_train20k_internvl3_8B_partial.json`

### Testing

- [OK] `wc -l save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`
- [OK] `ps -fp 41453,41464`
- [OK] `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`
- [OK] `tail -n 60` on both new restart logs showed `Model loaded` plus the first `[1/20000]` progress line

### Status

[IN PROGRESS] **Running**

### Next Steps

- Monitor the two restart logs until they end with:
  - `Done. Processed 20000 images ...`
  - `Saved attribute annotations to ...`
  - `Saved attribute CSV to ...`
  - `Removed partial checkpoint ...`
- After both finish, update `.handoff.md` again to replace the live PID references with completion status.
- Do not update `report.md` yet; the reruns are active and there is no new stable result to preserve.

---

## Session 16: Confirm Qwen3 train20k completion while InternVL3 continues

**Date**: 2026-04-13
**Task**: verify-qwen3-finish-after-restart

### Summary

Re-checked the fresh April 13, 2026 restart logs after several hours. The restarted Qwen3 `train20k` run completed successfully and rewrote the canonical output files, while the restarted InternVL3 job remained in progress.

### Main Changes

- Verified Qwen3 completion from the restart log:
  - `Done. Processed 20000 images in 401.4 minutes (6.69 hours)`
  - `Saved attribute annotations to save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
  - `Saved attribute CSV to save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv`
  - `Removed partial checkpoint save/attribute_coco_pose_gemini_train20k_qwen3_8B_partial.json`
- Verified the canonical Qwen3 output files now exist with fresh April 13 mtimes:
  - `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
  - `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv`
- Verified the canonical Qwen3 CSV now has `20,000` data rows (`20,001` lines including header).
- Verified InternVL3 is still running under `PID 41464`.
- Verified the latest InternVL3 restart log progress reached:
  - `[4700/20000] Processed 4700 images ...`
  - latest checkpoint save at `4500` images
- Verified the canonical InternVL3 CSV currently has `4,500` data rows (`4,501` lines including header) and the partial checkpoint remains active.

### Key Logs

- Qwen3 completed log:
  - `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_restart_gpu0_20260413_131326.log`
- InternVL3 active log:
  - `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_restart_gpu1_20260413_131327.log`

### Testing

- [OK] `tail -n 80 log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_restart_gpu0_20260413_131326.log`
- [OK] `rg -n "Checkpoint saved|Done\\. Processed|Saved attribute annotations|Saved attribute CSV|Removed partial" log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_restart_gpu0_20260413_131326.log | tail -n 10`
- [OK] `wc -l save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`
- [OK] `ps -fp 41464`

### Status

[IN PROGRESS] **Qwen3 completed; InternVL3 still running**

### Next Steps

- Keep monitoring `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_restart_gpu1_20260413_131327.log` until it reaches the same final save lines.
- Keep `report.md` unchanged for now; there is no new human-facing summary beyond the fact that the Qwen3 rerun finished successfully.

---

## Session 17: Remove `time_of_day=unknown` from the Gemini COCO pose corpus

**Date**: 2026-04-13
**Task**: remove-time-of-day-unknown-from-corpus

### Summary

After the user noticed `unknown` was still present in `save/corpus_coco_pose_gemini.json`, confirmed that the remaining `unknown` was in `time_of_day` rather than `weather`, then removed it from the canonical corpus.

### Main Changes

- Confirmed the corpus state before the edit:
  - `weather` already had no `unknown`
  - `time_of_day` still had:
    - `["morning", "afternoon", "evening", "night", "unknown"]`
- Updated `save/corpus_coco_pose_gemini.json` so `time_of_day` is now:
  - `["morning", "afternoon", "evening", "night"]`
- Verified the file content after the edit.
- Important runtime note:
  - the still-running InternVL3 restart process (`PID 41464`) was launched before this corpus edit
  - that live process has already loaded the old corpus into memory, so this file change does **not** retroactively change the currently running InternVL3 job
  - the already completed Qwen3 rerun also used the pre-edit corpus and may still contain `time_of_day=unknown` in its outputs

### Key File

- `save/corpus_coco_pose_gemini.json`

### Testing

- [OK] `sed -n '18,34p' save/corpus_coco_pose_gemini.json`
- [OK] `ps -fp 41464`

### Status

[OK] **Completed**

### Next Steps

- If the goal is only to fix the canonical corpus definition, no further action is needed.
- If the goal is to eliminate `time_of_day=unknown` from generated attribute outputs as well, rerun Qwen3 and restart InternVL3 after this corpus edit.

---

## Session 18: Archive `timeNA` outputs and relaunch Qwen3 / InternVL3 on GPU 1 and 3

**Date**: 2026-04-13
**Task**: relaunch-after-timeNA-corpus-cleanup

### Summary

The user requested that the old Qwen3 / InternVL3 outputs produced with the pre-cleanup `time_of_day=unknown` corpus be preserved under a `timeNA` suffix, that the active InternVL3 job be stopped, and that fresh canonical reruns start on GPU `1` and GPU `3`.

### Main Changes

- Stopped the active InternVL3 rerun:
  - original PID: `41464`
  - had been running on GPU `1`
- Archived the pre-cleanup outputs with `timeNA` suffixes.
- Archived the matching restart logs / pid files with `timeNA` suffixes.
- Relaunched fresh clean canonical jobs with the updated corpus that no longer contains `time_of_day=unknown`:
  - Qwen3 on GPU `1`
  - InternVL3 on GPU `3`
- Verified both new jobs:
  - are detached with `PPID 1`
  - are actively using GPU `1` / GPU `3`
  - loaded the COCO pose `train` split and reached the first `[1/20000]` progress line

### Archived `timeNA` Artifacts

- Qwen3:
  - `log_pose/NA/attribute_coco_pose_gemini_train20k_qwen3_8B_timeNA.json`
  - `log_pose/NA/attribute_coco_pose_gemini_train20k_qwen3_8B_timeNA.csv`
  - `log_pose/NA/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_restart_gpu0_20260413_131326_timeNA.log`
  - `log_pose/NA/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_restart_gpu0_20260413_131326_timeNA.pid`
- InternVL3:
  - `log_pose/NA/attribute_coco_pose_gemini_train20k_internvl3_8B_timeNA.json`
  - `log_pose/NA/attribute_coco_pose_gemini_train20k_internvl3_8B_timeNA.csv`
  - `log_pose/NA/attribute_coco_pose_gemini_train20k_internvl3_8B_timeNA_partial.json`
  - `log_pose/NA/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_restart_gpu1_20260413_131327_timeNA.log`
  - `log_pose/NA/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_restart_gpu1_20260413_131327_timeNA.pid`

### New Active Logs

- Qwen3 on GPU `1`:
  - `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_gpu1_20260413_201924.log`
  - `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_gpu1_20260413_201924.pid`
  - active PID: `132772`
- InternVL3 on GPU `3`:
  - `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_gpu3_20260413_201925.log`
  - `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_gpu3_20260413_201925.pid`
  - active PID: `132852`

### Canonical Output Paths Being Regenerated

- `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
- `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv`
- `save/attribute_coco_pose_gemini_train20k_qwen3_8B_partial.json`
- `save/attribute_coco_pose_gemini_train20k_internvl3_8B.json`
- `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`
- `save/attribute_coco_pose_gemini_train20k_internvl3_8B_partial.json`

### Testing

- [OK] `ps -fp 132772,132852`
- [OK] `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`
- [OK] `tail -n 60 log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_gpu1_20260413_201924.log`
- [OK] `tail -n 60 log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_gpu3_20260413_201925.log`

### Status

[IN PROGRESS] **Running on GPU 1 and GPU 3**

### Next Steps

- Wait for the first `500`-image checkpoint writes to recreate the canonical CSV / partial files under `save/`.
- Monitor both new logs until they end with the usual completion lines.

---

## Session 19: Restart stopped canonical COCO Gemini train20k reruns on GPU 0 and 1

**Date**: 2026-04-13
**Task**: restart-stopped-gemini-train20k-reruns

### Summary

The user noticed the latest clean canonical COCO Gemini train20k reruns had stopped again. I checked the saved state, confirmed both April 13 evening PIDs were dead, then restarted the work on GPU `0` and GPU `1` with resume enabled and verified the new detached jobs were progressing.

### Main Changes

- Confirmed the earlier clean canonical reruns from `2026-04-13 20:19 CDT` were no longer live:
  - Qwen3 log `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_gpu1_20260413_201924.log`
    - last saved checkpoint at `4000` completed images
  - InternVL3 log `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_gpu3_20260413_201925.log`
    - last visible progress at `800`, but only the `500`-image checkpoint had been written
- Verified the canonical resume files before relaunch:
  - `save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv` had `4000` data rows
  - `save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv` had `500` data rows
- First attempted the requested GPU `0` / `1` relaunch with:
  - Qwen3 on GPU `0`
  - InternVL3 on GPU `1`
  - explicit `--resume`
  - explicit canonical output / CSV / partial paths
- That first attempt failed during model load because those two GPUs still had competing processes:
  - ignore `log_pose/NA/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_resume_gpu0_20260413_233109.log`
  - ignore `log_pose/NA/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu1_20260413_233109.log`
- Killed the failed OOM workers and rechecked GPU occupancy.
- After GPU `0` became free, relaunched with a more stable mapping while still using GPU `0` and GPU `1`:
  - InternVL3 on GPU `0`
  - Qwen3 on GPU `1`
- Set `CONDA_DEFAULT_ENV=qwen` before launch so `utils/assign_attribute.py` would not spawn an extra `conda run` wrapper inside the detached job.
- Verified the new detached jobs:
  - have `PPID 1`
  - are using the intended GPUs via `nvidia-smi`
  - restored from the existing canonical CSV / partial checkpoints
  - advanced past startup into resumed processing

### Key Outputs

- Failed first restart attempt:
  - `log_pose/NA/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_resume_gpu0_20260413_233109.log`
  - `log_pose/NA/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu1_20260413_233109.log`
- Current live reruns:
  - InternVL3:
    - log `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu0_20260413_233306.log`
    - pid `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu0_20260413_233306.pid`
    - live PID `72034`
  - Qwen3:
    - log `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_resume_gpu1_20260413_233306.log`
    - pid `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_resume_gpu1_20260413_233306.pid`
    - live PID `72037`

### Runtime Notes

- Verified resume points from the new logs:
  - InternVL3: `Resuming from CSV checkpoint: 500 images already processed`
  - Qwen3: `Resuming from CSV checkpoint: 4000 images already processed`
- Verified early resumed progress:
  - InternVL3 reached `[1/19500]`
  - Qwen3 reached `[100/16000]`
- Because checkpoint saving is every `500` resumed images, the canonical CSV / partial files under `save/` still had their old timestamps at the time of verification:
  - Qwen3 next expected save: overall image `4500`
  - InternVL3 next expected save: overall image `1000`

### Testing

- [OK] `ps -fp 132772 132852`
- [OK] `tail -n 60 log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_gpu1_20260413_201924.log`
- [OK] `tail -n 60 log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_gpu3_20260413_201925.log`
- [OK] `wc -l save/attribute_coco_pose_gemini_train20k_qwen3_8B.csv save/attribute_coco_pose_gemini_train20k_internvl3_8B.csv`
- [OK] `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`
- [OK] `ps -fp 72034 72037`
- [OK] `rg -n "\\[100/16000\\]|Checkpoint saved" log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_resume_gpu1_20260413_233306.log`
- [OK] `rg -n "\\[100/19500\\]|Checkpoint saved" log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu0_20260413_233306.log`

### Status

[IN PROGRESS] **Running on GPU 0 and GPU 1**

### Next Steps

- Keep monitoring:
  - `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train20k_resume_gpu1_20260413_233306.log`
  - `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_train20k_resume_gpu0_20260413_233306.log`
- Wait for the next resumed checkpoint writes:
  - Qwen3 overall `4500`
  - InternVL3 overall `1000`
- Once the canonical CSV / partial files advance, continue monitoring until both runs reach the usual final save lines.

---

## Session 20: Add `--start_at` and queue Qwen3 val then train-after-20k on GPU 1

**Date**: 2026-04-14
**Task**: qwen3-val-then-train-after20k

### Summary

The user asked to run Qwen3 `val` on GPU `1`, then continue with the training images after the earlier `train20k` subset. I interpreted `training2k` as `train20k`, added a minimal start-offset option to `utils/assign_attribute.py`, and launched a sequential GPU `1` wrapper that first reruns canonical Qwen3 `val` and then automatically starts the remaining train images after index `20000`.

### Main Changes

- Confirmed the current canonical Qwen3 `val` artifacts were incomplete:
  - `save/attribute_coco_pose_gemini_qwen3_8B.csv` had only `500` data rows (`501` lines including header)
  - this means the canonical `val` output needed to be rerun, not reused
- Confirmed `utils/assign_attribute.py` supported `--stop_at` but had no way to skip the first part of a split.
- Updated `utils/assign_attribute.py`:
  - added `--start_at`
  - applied the offset before `--stop_at`
  - included the start offset in the auto-generated tag when output paths are not provided explicitly
- Chose explicit new output paths for the post-20k train continuation so the canonical `train20k` files are left intact:
  - `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.json`
  - `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.csv`
  - `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B_partial.json`
- Launched a detached sequential wrapper on GPU `1`:
  - stage 1: rerun canonical Qwen3 `val`
  - stage 2: if stage 1 succeeds, run Qwen3 `train` with `--start_at 20000`
- Kept the currently active InternVL3 canonical `train20k` rerun on GPU `0` unchanged.

### Key Outputs

- Code change:
  - `utils/assign_attribute.py`
- Wrapper and logs:
  - wrapper log: `log_pose/NA/qwen3_val_then_train_after20k_gpu1_20260414_134529.log`
  - wrapper pid file: `log_pose/NA/qwen3_val_then_train_after20k_gpu1_20260414_134529.pid`
  - wrapper PID: `339247`
  - val log: `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_val_gpu1_20260414_134529.log`
  - queued train-after-20k log: `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train_after20k_gpu1_20260414_134529.log`
- Regenerated canonical val targets:
  - `save/attribute_coco_pose_gemini_qwen3_8B.json`
  - `save/attribute_coco_pose_gemini_qwen3_8B.csv`
  - `save/attribute_coco_pose_gemini_qwen3_8B_partial.json`
- New train-after-20k targets:
  - `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.json`
  - `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.csv`
  - `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B_partial.json`

### Runtime Notes

- Interpretation note:
  - the user wording was `training2k`
  - because the repo's active workflow centers on the canonical `train20k` subset, I interpreted this as "the train images after the first 20,000 valid train2017 images"
- Verified the new val job startup:
  - `Model loaded.`
  - `Loaded COCO pose split 'val' ... 2346 valid images`
  - `[1/2346]`
- GPU occupancy after launch:
  - GPU `0`: active InternVL3 train20k rerun
  - GPU `1`: active Qwen3 val rerun

### Testing

- [OK] `python3 -m py_compile utils/assign_attribute.py`
- [OK] `ps -fp 339247`
- [OK] `tail -n 60 log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_val_gpu1_20260414_134529.log`
- [OK] `nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader`

### Status

[IN PROGRESS] **Qwen3 val running on GPU 1; train-after-20k queued next**

### Next Steps

- Monitor `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_val_gpu1_20260414_134529.log` until it reaches the usual final save lines.
- Monitor `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train_after20k_gpu3_20260414_134834.log`.
- Once the train-after-20k run finishes, decide whether to merge the first `20,000` rows and the after-`20,000` rows into a single full-train Qwen3 artifact or keep them as two explicit shards.

### Follow-up Runtime Update

- The user clarified that `training2k` did mean `train20k`.
- The initial sequential wrapper on GPU `1` was then adjusted to match the desired GPU placement:
  - stopped wrapper PID `339247`
  - the live `val` child survived and continued as PID `339278` with `PPID 1`
- Launched `train-after-20k` separately on GPU `3`:
  - log `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_train_after20k_gpu3_20260414_134834.log`
  - pid `342164`
- Verified the new GPU `3` run printed:
  - `Skipping the first 20000 images (--start_at 20000); 36599 remain`
  - `[500/36599]`
  - `Checkpoint saved to save/attribute_coco_pose_gemini_train_after20k_qwen3_8B_partial.json (500 images done)`
- Verified the first saved rows of `save/attribute_coco_pose_gemini_train_after20k_qwen3_8B.csv` begin with:
  - `206591`
  - `206596`
  - `206602`
  - `206606`
  - `206613`
- Cross-checked against the tail of the earlier canonical `train20k` Qwen3 CSV:
  - `206539`
  - `206560`
  - `206583`
- Therefore the new shard is correctly starting after the first `20,000` valid train images rather than repeating the old `train20k` prefix.

### Follow-up Runtime Update 2

- The Qwen3 canonical `val` rerun on GPU `1` completed successfully on April 14, 2026:
  - log `log_pose/assign_attribute_coco_pose_qwen3_8B_gemini_val_gpu1_20260414_134529.log`
  - final saves:
    - `save/attribute_coco_pose_gemini_qwen3_8B.json`
    - `save/attribute_coco_pose_gemini_qwen3_8B.csv`
- Added convenience alias copies with an explicit split suffix while keeping the canonical filenames unchanged for existing code paths:
  - `save/attribute_coco_pose_gemini_qwen3_8B_val.json`
  - `save/attribute_coco_pose_gemini_qwen3_8B_val.csv`
- After that completion, launched a fresh InternVL3 canonical `val` rerun on GPU `1`:
  - log `log_pose/assign_attribute_coco_pose_internvl3_8B_gemini_val_gpu1_20260414_143656.log`
  - pid `383636`
- Verified the new InternVL3 val startup:
  - `Model loaded.`
  - `Loaded COCO pose split 'val' ... 2346 valid images`
  - `[1/2346]`
- GPU placement after this update:
  - GPU `0`: InternVL3 canonical train20k rerun
  - GPU `1`: InternVL3 canonical val rerun
  - GPU `3`: Qwen3 train-after-20k rerun

---

## Session 21: `stat_pose_estimation.py` suffix robustness and train-only fallback

**Date**: 2026-04-14
**Task**: pose-slice-stats-suffix-and-train-only

### Summary

Updated `utils/stat_pose_estimation.py` so it tolerates explicit split suffixes such as `_val`, falls back between suffixed and unsuffixed attribute filenames, and no longer aborts when no eval-side attribute coverage is available. In the no-eval-coverage case, the script now emits train-side distribution rows with eval-side fields left blank.

### Main Changes

- Added suffix-aware attribute-path resolution:
  - `split -> suffix` mapping now treats both `val` and `val_full` as `_val`
  - if the requested attribute path does not exist, the script now also tries the corresponding suffixed or unsuffixed sibling path
  - this applies to both `--attribute-file` and `--train-attribute-file`
- Updated slice-row sorting so rows with blank eval metrics do not crash sort-by-`ap`, sort-by-`ap_delta`, or sort-by-`images`.
- Changed CSV writing behavior for missing fields:
  - `None` is now written as an empty cell instead of the literal string `NA`
- Reworked the no-eval-coverage path:
  - if no eval images are covered but train-side attributes can still be recovered, the script no longer returns an empty NA summary
  - it now writes `overall` with blank eval metrics plus `coverage_images=0`
  - it emits per-value rows for train-supported values that meet `--min-images`, with train counts / shares filled and eval-side fields blank
  - it sets JSON `status: "train_only"` with an explanatory `reason`
- Preserved the real unavailable path for harder failures:
  - if neither eval coverage nor recoverable train-side distribution exists, the script still writes the older unavailable summary

### Validation

- [OK] `python3 -m py_compile utils/stat_pose_estimation.py`
- [OK] Verified suffix fallback in the `ad` environment:
  - command requested missing `save/attribute_coco_pose_gemini_train20k_qwen3_8B_train.json`
  - script printed fallback to `save/attribute_coco_pose_gemini_train20k_qwen3_8B.json`
  - smoke outputs:
    - `/tmp/stat_pose_suffix_smoke.json`
    - `/tmp/stat_pose_suffix_smoke.csv`
- [OK] Verified train-only fallback in the `ad` environment:
  - command used `--attribute-file save/attribute_coco_pose_gemini_train20k_qwen3_8B.json --attribute-split train --eval-split val`
  - no eval coverage was found, but the script still wrote train-side `weather` rows
  - smoke outputs:
    - `/tmp/stat_pose_train_only_smoke.json`
    - `/tmp/stat_pose_train_only_smoke.csv`
  - confirmed JSON rows contain:
    - blank eval metrics
    - `train_count` / `train_share` values
    - `status: "train_only"`
  - confirmed CSV output uses empty cells rather than `NA`

### Important Notes

- The `rare_case` rule in this script still depends on eval-side AP being below the overall AP as well as train-side rarity.
- Therefore, in train-only mode the script leaves `is_failure_case` and `failure_type` blank instead of labeling slices as rare / hard / shift without eval evidence.
- While testing, a deliberate mis-specified run with `--attribute-split val` against the full `2346`-row `_val` file failed with the expected length-mismatch error, which confirms the split-ordering checks are still active.

### Status

[OK] **Completed**
