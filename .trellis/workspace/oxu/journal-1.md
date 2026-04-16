# Journal - oxu (Part 1)

> AI development session journal
> Started: 2026-02-24

---

## Session 65: Summarize completed CelebA attribute ratio-0.1 probe

**Date**: 2026-04-16
**Task**: celeba-attribute-ratio0p1-results-summary

### Summary

The user reported that the three CelebA attribute-AL hyperparameter runs at `ratio=0.1` appeared finished and asked for the results. Verified that all three launched workers had exited, extracted the final held-out test accuracies, selection times, and wall times from their logs, compared them against the completed CelebA `ratio=0.1` baselines, and recorded the stable outcome in `report.md`.

### Main Changes

- Updated:
  - `report.md`

### Main Findings

- Completed attribute configs (`E=False` for all):
  - `baseline_def_e0_r001_p5_cinf`
    - `R=0.01`, `P=5.0`, `C=inf`
    - held-out test accuracy `0.9059`
    - AL selection time `0.72 s`
    - end-to-end wall time `237.61 s`
    - log:
      - `logs_celeba/attr_ratio0.1_baseline_def_e0_r001_p5_cinf_gpu0_20260416_140026.log`
  - `soft_e0_r01_p5_c15`
    - `R=0.1`, `P=5.0`, `C=15.0`
    - held-out test accuracy `0.9055`
    - AL selection time `0.70 s`
    - end-to-end wall time `236.93 s`
    - log:
      - `logs_celeba/attr_ratio0.1_soft_e0_r01_p5_c15_gpu1_20260416_140026.log`
  - `aggro_e0_r08_p10_c4`
    - `R=0.8`, `P=10.0`, `C=4.0`
    - held-out test accuracy `0.9047`
    - AL selection time `0.71 s`
    - end-to-end wall time `236.35 s`
    - log:
      - `logs_celeba/attr_ratio0.1_aggro_e0_r08_p10_c4_gpu2_20260416_140026.log`
- Ranking within this probe:
  - best:
    - `baseline_def_e0_r001_p5_cinf`
    - `90.59%`
  - second:
    - `soft_e0_r01_p5_c15`
    - `90.55%`
  - third:
    - `aggro_e0_r08_p10_c4`
    - `90.47%`
- Comparison against completed CelebA `ratio=0.1` baselines already in the report:
  - `random`: `91.15%`
  - `badge`: `91.12%`
  - `stratified`: `90.95%`
  - `coreset`: `90.29%`
- Interpretation:
  - the best attribute configuration barely matches the manually reported base reference (`90.57%`)
  - none of the three attribute configurations beat the completed `random`, `badge`, or `stratified` baselines at `ratio=0.1`
  - runtime is excellent, but current accuracy is not yet competitive

### Validation

- [OK] `ps -fp 3040154,3040157,3040160` showed the three launch PIDs had exited
- [OK] inspected the final metric blocks and tails of all three logs
- [OK] `git diff --check -- report.md .trellis/workspace/oxu/journal-1.md .handoff.md`

### Important Notes

- These runs completed cleanly enough to use their `Model Test ACC` lines as the current CelebA attribute-probe outcome.
- The existing deprecation warning in `alpred.py:1705` still appears during prediction saving, but it did not prevent completion.
- The next sensible follow-up is not a wider ratio sweep with the aggressive setting; if the user wants to continue CelebA attribute experiments, start from the conservative `R=0.01, P=5.0, C=inf, E=False` setting or revisit the attribute scoring design.

### Status

[OK] **Completed**

---

## Session 64: Launch CelebA attribute-AL ratio-0.1 RPC sweep

**Date**: 2026-04-16
**Task**: celeba-attribute-ratio0p1-rpc-launch

### Summary

The user asked to start CelebA attribute-driven Active Learning experiments in `alpred.py`, matching the ImageNet workflow but first sweeping a few hyperparameter settings at `ratio=0.1`, with entropy fixed off and three `R/P/C` groups run in parallel on GPUs `0/1/2`. Audited the ImageNet attribute sweep outcomes, inspected the CelebA attribute metadata and selector code, found that the CelebA metadata keys use spaced names (`hair color`, `skin`, `emotion`) so the ImageNet-style auto-inferred synthetic counts would not trigger, created a dedicated launch script, and started three background resume-from-base runs.

### Main Changes

- Added:
  - `scripts/run_celeba_attribute_ratio0.1_rpc.sh`
- Launched:
  - GPU `0`, `baseline_def_e0_r001_p5_cinf`
  - GPU `1`, `soft_e0_r01_p5_c15`
  - GPU `2`, `aggro_e0_r08_p10_c4`

### Main Findings

- ImageNet reference used to choose the grid:
  - `ratio=0.1` best completed attribute sweep log was:
    - `logs/logs_imagenet/sweep_attr_baseline_def_ratio0.1.log`
    - held-out test accuracy `0.7300`
  - `ratio=0.3` best completed attribute sweep log was:
    - `logs/logs_imagenet/sweep_attr_aggro_exp_p10_c4_ratio0.3.log`
    - held-out test accuracy `0.7415`
- Chosen CelebA `R/P/C` groups with `E=False`:
  - conservative strict balance:
    - `R=0.01`, `P=5.0`, `C=inf`
  - soft conservative:
    - `R=0.1`, `P=5.0`, `C=15.0`
  - aggressive:
    - `R=0.8`, `P=10.0`, `C=4.0`
- CelebA-specific selector caveat found during inspection:
  - `save/attribute.json` uses keys:
    - `hair color`
    - `skin`
    - `emotion`
  - therefore `infer_generated_attribute_counts()` does not automatically map the string `redhair_brownskin_sademotion` onto those keys
  - the launched runs therefore pass explicit:
    - `--al_attribute_focus "hair color,skin,emotion"`
    - `--al_attribute_extra_counts "hair color:red=1000,skin:brown=1000,emotion:sad=1000"`
- Quick selection-only CelebA probe before launch showed strong enrichment over random at `ratio=0.1`:
  - random `8000`-sample reference:
    - `red`: `258`
    - `brown skin`: `2415`
    - `sad`: `363`
  - selected `8000`-sample ranges across the three configs:
    - `red`: `979` to `1016`
    - `brown skin`: `3430` to `3501`
    - `sad`: `771` to `823`
- Active runs launched from:
  - base checkpoint:
    - `weights/model_celeba_resnet_base.pth`
  - launcher:
    - `scripts/run_celeba_attribute_ratio0.1_rpc.sh`
- Live run details:
  - GPU `0`
    - PID `3040154`
    - log `logs_celeba/attr_ratio0.1_baseline_def_e0_r001_p5_cinf_gpu0_20260416_140026.log`
    - pid file `logs_celeba/pids/attr_ratio0.1_baseline_def_e0_r001_p5_cinf_gpu0_20260416_140026.pid`
  - GPU `1`
    - PID `3040157`
    - log `logs_celeba/attr_ratio0.1_soft_e0_r01_p5_c15_gpu1_20260416_140026.log`
    - pid file `logs_celeba/pids/attr_ratio0.1_soft_e0_r01_p5_c15_gpu1_20260416_140026.pid`
  - GPU `2`
    - PID `3040160`
    - log `logs_celeba/attr_ratio0.1_aggro_e0_r08_p10_c4_gpu2_20260416_140026.log`
    - pid file `logs_celeba/pids/attr_ratio0.1_aggro_e0_r08_p10_c4_gpu2_20260416_140026.pid`
- Immediate post-launch status:
  - all three logs already passed attribute selection
  - each log contains:
    - `CelebA: Combined 8000 selected samples with 1000 generated images using Active Learning`
    - `Resumed model from /people/cs/o/oxx220000/safefix/weights/model_celeba_resnet_base.pth ...`

### Validation

- [OK] `bash -n scripts/run_celeba_attribute_ratio0.1_rpc.sh`
- [OK] `bash scripts/run_celeba_attribute_ratio0.1_rpc.sh --dry-run`
- [OK] verified launched PIDs with:
  - `ps -fp 3040154,3040157,3040160`
- [OK] tailed all three logs and confirmed selection finished plus resumed training startup

### Important Notes

- `report.md` was not updated because these runs are still in progress and there are no stable held-out test results yet.
- The new launcher fixes a CelebA-specific metadata mismatch that would otherwise make the frequency-penalty term ignore the injected `1000` synthetic `red/brown/sad` samples.
- Recommended follow-up:
  - monitor the three logs for `Held-out test accuracy`
  - pick the best `R/P/C` setting at `ratio=0.1`
  - only then extend the attribute sweep to larger ratios

### Status

[OK] **Completed**

---

## Session 62: Summarize completed CelebA baselines in report

**Date**: 2026-04-16
**Task**: celeba-baseline-report-summary

### Summary

The user said that most previously run CelebA baselines were already finished and asked to summarize them in `report.md`. Audited the completed non-hierarchical CelebA resume logs, extracted the held-out test accuracies and selection times, chose canonical rerun logs where patched large-job selectors replaced stalled originals, and added a compact results section plus observations to the report.

### Main Changes

- Updated:
  - `report.md`

### Main Findings

- Completed and summarized in the report:
  - `random` ratios `0.1`, `0.3`, `0.5`, `0.7`
  - `stratified` ratios `0.1`, `0.3`, `0.5`, `0.7`
  - `coreset` ratios `0.1`, `0.3`, `0.5`, patched rerun `0.7`
  - `badge` ratio `0.1`, patched reruns `0.3`, `0.5`, `0.7`
- Best completed resume result:
  - `badge` at ratio `0.7`
  - held-out test accuracy `0.9131`
  - canonical log:
    - `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.7_gpu0_20260416_122047.log`
- Base checkpoint remains slightly better:
  - base held-out test accuracy `0.9140`
- Runtime finding preserved in the report:
  - pre-patch `coreset` and `badge` selection became very expensive at higher ratios
  - patched large-job reruns reduced selector time sharply while keeping the sweep practical

### Validation

- [OK] inspected final tails and metrics from the canonical logs listed in the new report table
- [OK] confirmed `report.md` now contains the new CelebA resume-baseline summary section

### Important Notes

- `hierarchical` remains deferred and was intentionally not summarized as a completed baseline set.
- `.handoff.md` was not updated because the active baton-pass state did not change; this step only documented already completed results.

### Status

[OK] **Completed**

---

## Session 63: Reformat CelebA report tables and time display

**Date**: 2026-04-16
**Task**: celeba-report-reformat-and-manual-base-acc

### Summary

The user asked to manually change the CelebA ResNet base test accuracy in `report.md` to `90.57`, to stop reporting long durations in seconds when they exceed ten minutes, and to split the CelebA baseline summary into four separate tables by ratio (`0.1`, `0.3`, `0.5`, `0.7`). Updated the report accordingly and recorded the new formatting convention in this journal entry.

### Main Changes

- Updated:
  - `report.md`

### Main Findings

- The CelebA base held-out test accuracy is now manually reported as:
  - `90.57%`
- The CelebA resume-baseline summary is now organized as four separate ratio-specific tables:
  - `ratio = 0.1`
  - `ratio = 0.3`
  - `ratio = 0.5`
  - `ratio = 0.7`
- The CelebA baseline table values were reformatted from decimal accuracies to percentage display for readability.
- Time-display convention applied in the updated report sections:
  - durations longer than ten minutes are written as `X min` or `Y h Z min`
  - shorter durations keep second-level display

### Validation

- [OK] visually checked the updated top report sections after the patch
- [OK] `git diff --check -- report.md .trellis/workspace/oxu/journal-1.md`

### Important Notes

- This step follows the user's manual reporting preference for the CelebA base test accuracy rather than recomputing from the underlying log.
- `.handoff.md` was not updated because the active baton-pass state did not change.

### Status

[OK] **Completed**

---

## Session 58: Diagnose stuck CelebA AL baselines and stop the queue

**Date**: 2026-04-16
**Task**: celeba-al-stall-diagnosis-and-kill

### Summary

The user reported that the ongoing CelebA resume baseline sweep appeared stuck around the `hierarchical` stage and asked to find the issue and kill the affected jobs. Inspected the active controller, worker logs, and AL strategy implementations, confirmed that multiple Active Learning strategies were stalled in the selection phase before training started, and terminated the active controller plus the stuck workers.

### Main Changes

- Stopped active controller:
  - PID `2728053`
- Stopped stuck workers:
  - `coreset`, ratio `0.7`, PID `2754132`
  - `hierarchical`, ratio `0.1`, PID `2776030`
  - `badge`, ratio `0.3`, PID `2779384`
  - `badge`, ratio `0.5`, PID `2820373`

### Main Findings

- Completed successfully before the stall:
  - `random` ratios `0.1`, `0.3`, `0.5`, `0.7`
  - `stratified` ratios `0.1`, `0.3`, `0.5`, `0.7`
  - `coreset` ratios `0.1`, `0.3`, `0.5`
  - `badge` ratio `0.1`
- The queue reached `hierarchical` for ratio `0.1` at:
  - `logs_celeba/run_all_resume_baselines_20260416_040048.log`
  - launch line for worker PID `2776030`
- The stuck `hierarchical` worker log never advanced beyond:
  - `Current time: 0416071719`
  - `Selected 1000 filenames for sampling`
  - `Added 1000 generated redhair_brownskin_sademotion images...`
  - file:
    - `logs_celeba/resume_lr1e-5_epoch15_hierarchical_resnet_celeba_1000_ratio0.1.log`
- The later stuck `badge` workers showed the same symptom:
  - `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.3.log`
  - `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.5.log`
  - both stopped after the same three startup lines with no selection or epoch output
- The active `coreset` ratio `0.7` worker was not deadlocked but still impractically expensive:
  - log:
    - `logs_celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.7.log`
  - it had only progressed through the core-set progress bar up to:
    - `47600/56000`
  - and was still consuming a full CPU core
- Root-cause diagnosis from code inspection:
  - `utils/active_learning.py` `CoreSetStrategy.greedy_k_center()` performs a full `euclidean_distances()` pass over all unlabeled samples on every selection step
  - for CelebA ratio `0.7`, this means selecting `56000` of `80000` samples, which is prohibitively expensive
  - `utils/active_learning.py` `BadgeStrategy._kmeanspp_seeding()` has the same repeated full-distance update structure for each selected point
  - `utils/hierarchical_al.py` adds another expensive layer by repeatedly scanning all samples for slice membership and then running greedy within-slice distance-based selection
- This is therefore an algorithmic scalability issue in the AL selectors, not a GPU assignment problem and not a training deadlock

### Validation

- [OK] inspected active controller tail:
  - `logs_celeba/run_all_resume_baselines_20260416_040048.log`
- [OK] inspected stuck worker logs listed above
- [OK] verified the terminated PIDs no longer appear in:
  - `ps -fp 2728053,2754132,2776030,2779384,2820373`

### Important Notes

- `badge` ratio `0.1` did complete, but it still recorded a very large AL selection time:
  - `Total Active Learning selection time: 6695.79 seconds`
  - log:
    - `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.1.log`
- That long-but-finished ratio `0.1` run is consistent with the same scaling problem that caused the larger-ratio `badge` runs to appear hung.
- `report.md` was not updated because this step diagnosed and stopped an incomplete sweep; there are no new stable report-level results.

### Status

[OK] **Completed**

---

## Session 59: Relaunch remaining non-BADGE/Hierarchical run and patch BADGE scaling

**Date**: 2026-04-16
**Task**: celeba-relaunch-coreset-and-rerun-badge03

### Summary

The user asked to first restart any unfinished CelebA baselines excluding `badge` and `hierarchical`, then modify the code and rerun `badge` at ratio `0.3`. Confirmed that the only unfinished non-`badge`/non-`hierarchical` job was `coreset` at ratio `0.7`, relaunched it, patched the BADGE strategy to avoid the quadratic-style large-run kmeans++ selection path, and launched a fresh `badge` `0.3` rerun.

### Main Changes

- Updated:
  - `utils/active_learning.py`
- Relaunched unfinished non-`badge`/non-`hierarchical` run:
  - `coreset`, ratio `0.7`, GPU `3`
  - PID `2873191`
  - log `logs_celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.7_rerun.log`
  - pid file `logs_celeba/pids/resume_lr1e-5_epoch15_coreset_ratio0.7_rerun.pid`
- Launched patched BADGE rerun:
  - `badge`, ratio `0.3`, GPU `1`
  - PID `2875339`
  - log `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.3_rerun_20260416_115621.log`
  - pid file `logs_celeba/pids/resume_lr1e-5_epoch15_badge_ratio0.3_rerun_20260416_115621.pid`

### Main Findings

- Completion audit before relaunch:
  - completed:
    - all `random`
    - all `stratified`
    - `coreset` ratios `0.1`, `0.3`, `0.5`
    - `badge` ratio `0.1`
  - unfinished excluding `badge` and `hierarchical`:
    - only `coreset` ratio `0.7`
- BADGE patch details:
  - kept exact BADGE kmeans++ selection for smaller jobs
  - added a fast path for larger jobs in `BadgeStrategy.select()`
  - the fast path switches to gradient-norm weighted sampling without replacement instead of repeated full-distance kmeans++ updates
  - this change specifically targets the CelebA large-ratio stall pattern where exact BADGE selection was too slow to finish
- Immediate post-launch status:
  - `coreset 0.7` rerun log already reached:
    - `Core-set selection: choosing 56000 from 80000 samples...`
  - patched `badge 0.3` rerun is alive and has written the initial dataset-setup lines to its log
  - because worker logs run with `--verbose False`, the BADGE selection stage still does not print intermediate progress, so success should be checked later from the same rerun log

### Validation

- [OK] `/people/cs/o/oxx220000/.conda/envs/ad/bin/python -m py_compile utils/active_learning.py`
- [OK] `git diff --check -- utils/active_learning.py .trellis/workspace/oxu/journal-1.md .handoff.md`
- [OK] verified relaunch PIDs:
  - `ps -fp 2873191`
  - `ps -fp 2875339`

### Important Notes

- The BADGE patch is an approximation for large jobs, not the original exact BADGE kmeans++ batch seeding.
- This was chosen intentionally so the CelebA ratio `0.3` rerun can finish in practical time instead of stalling in selection.
- `report.md` was not updated because these are in-progress reruns, not stable final results.

### Status

[OK] **Completed**

---

## Session 60: Launch BADGE ratios 0.5 and 0.7 on GPU 0 and confirm hierarchical status

**Date**: 2026-04-16
**Task**: celeba-badge-gpu0-05-07

### Summary

The user reported that BADGE was now able to run and asked to launch ratios `0.5` and `0.7` on GPU `0`, then asked whether the hierarchical code had also been modified. Started BADGE `0.5` on GPU `0`, then created a lightweight controller that waits for that run to finish before starting BADGE `0.7` on the same GPU to avoid overlapping two training jobs on one card. Hierarchical code was not modified in this step.

### Main Changes

- Active BADGE run on GPU `0`:
  - ratio `0.5`
  - PID `2898496`
  - log `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.5_gpu0_20260416_120554.log`
- Waiting controller for the follow-up BADGE run:
  - waits for PID `2898496`
  - controller PID `2901712`
  - controller log `logs_celeba/run_badge_gpu0_ratio0.7_after_0.5_20260416_120710.log`
  - controller pid file `logs_celeba/pids/run_badge_gpu0_ratio0.7_after_0.5_20260416_120710.pid`
- Planned next run from the controller:
  - BADGE ratio `0.7`
  - GPU `0`
  - output log path will be emitted by the controller after launch

### Main Findings

- The active GPU `0` BADGE `0.5` run has passed dataset setup and is alive.
- The `0.7` run is intentionally queued rather than launched concurrently so both runs use GPU `0` as requested without stepping on each other.
- Hierarchical status:
  - `utils/hierarchical_al.py` has **not** been modified yet
  - only `utils/active_learning.py` BADGE logic was changed previously

### Validation

- [OK] verified active BADGE `0.5` process:
  - `ps -fp 2898496`
- [OK] verified queued controller:
  - `ps -fp 2901712`
- [OK] inspected active BADGE `0.5` log:
  - `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.5_gpu0_20260416_120554.log`

### Important Notes

- The GPU `2` verbose BADGE debug run remains separate from these GPU `0` runs.
- `report.md` was not updated because these runs are still in progress.

### Status

[OK] **Completed**

---

## Session 61: Point local HiBug generated-data paths to shared /data mount

**Date**: 2026-04-16
**Task**: hibug-generated-data-relocation

### Summary

The user asked to move four HiBug-generated dataset folders from `./data/` to `/data/oxx220000/hibug_generated`, then update any code that references the old paths. In this workspace, those four folders were already present under `/data/oxx220000/hibug_generated/` and were not present as real directories under `./data/`. To keep the repo-local relative paths working (and avoid scattered hard-coded absolute paths), created symlinks under `data/` that point to the shared `/data` mount.

### Main Changes

- Added symlinks:
  - `data/hibug_redhair` -> `/data/oxx220000/hibug_generated/hibug_redhair`
  - `data/hibug_brownskin` -> `/data/oxx220000/hibug_generated/hibug_brownskin`
  - `data/hibug_redhair_brownskin` -> `/data/oxx220000/hibug_generated/hibug_redhair_brownskin`
  - `data/hibug_yellowhair_04300231` -> `/data/oxx220000/hibug_generated/hibug_yellowhair_04300231`

### Reference Search

- No tracked code/config in this repo referenced these four paths directly (search for `data/hibug_*` and `./data/hibug_*` returned no matches).
- The only mentions of `hibug_redhair` / `hibug_redhair_brownskin` were in `scripts/nvidia_draft.sh` (draft commands), and those lines were not used by the current training pipelines.

### Validation

- [OK] verified symlinks exist via `ls -la data/` and point to `/data/oxx220000/hibug_generated/...`

### Status

[OK] **Completed**

### Follow-up

- Updated `scripts/nvidia_draft.sh` to pass `--dir data/hibug_*` (so it follows the new `data/hibug_*` symlinks instead of relying on CWD-relative `hibug_*`).


## Session 61: Stop stuck CoreSet on GPU 3 and patch large-run CoreSet scaling

**Date**: 2026-04-16
**Task**: celeba-coreset-kill-and-patch

### Summary

The user observed that the GPU `3` CelebA `coreset` rerun also appeared stuck and asked to handle both stopping it and modifying the code. Confirmed that the active run was still pinned in the `Core-set selection: choosing 56000 from 80000 samples...` stage with no further log progress, stopped it, and patched `CoreSetStrategy` so large jobs use a fast approximate selection path instead of the exact repeated full-distance greedy k-center loop.

### Main Changes

- Updated:
  - `utils/active_learning.py`
- Stopped stuck run:
  - `coreset`, ratio `0.7`, GPU `3`
  - former PID `2873191`
  - log `logs_celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.7_rerun.log`

### Main Findings

- Before termination, the stuck run was still:
  - consuming ~`99%` CPU
  - using GPU `3`
  - with log frozen at:
    - `Core-set selection: choosing 56000 from 80000 samples...`
    - progress `0/56000`
- Root cause matched the earlier diagnosis:
  - `CoreSetStrategy.greedy_k_center()` updated distances against the full unlabeled pool once per selected sample
  - for CelebA ratio `0.7`, this means `56000` sequential full-pool updates, which is not practical
- New CoreSet behavior:
  - smaller jobs still use the original exact greedy k-center path
  - larger jobs now switch to approximate weighted sampling without replacement using the initial distance-to-labeled scores as weights
  - all patched call sites now pass `random_seed` through so the approximation remains reproducible

### Validation

- [OK] confirmed the stuck process no longer exists:
  - `ps -fp 2873191`
- [OK] `/people/cs/o/oxx220000/.conda/envs/ad/bin/python -m py_compile utils/active_learning.py`
- [OK] `git diff --check -- utils/active_learning.py .trellis/workspace/oxu/journal-1.md .handoff.md`

### Important Notes

- This CoreSet patch is an approximation for larger jobs, not the original exact k-center algorithm.
- The change is intentional so CelebA `coreset` high-ratio runs can complete in practical time.
- `report.md` was not updated because no new stable finished run was produced in this step.

### Status

[OK] **Completed**

---

## Session 62: Relaunch CoreSet on GPU 3 and start hierarchical scalability patch

**Date**: 2026-04-16
**Task**: celeba-rerun-coreset-gpu3-and-patch-hierarchical

### Summary

The user asked to relaunch the GPU `3` CoreSet run and to start modifying the hierarchical strategy. Relaunched CoreSet ratio `0.7` on GPU `3` using the newly patched CoreSet code and confirmed that it moved past the previous selection stall. Also patched hierarchical selection to reduce the expensive within-slice greedy selection cost and avoid repeated full-dataset slice-membership recomputation.

### Main Changes

- Updated:
  - `utils/hierarchical_al.py`
- Relaunched:
  - `coreset`, ratio `0.7`, GPU `3`
  - PID `2910596`
  - log `logs_celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.7_gpu3_rerun_20260416_121045.log`
  - pid file `logs_celeba/pids/resume_lr1e-5_epoch15_coreset_ratio0.7_gpu3_rerun_20260416_121045.pid`

### Main Findings

- The relaunched CoreSet run no longer stalls at `0/56000`:
  - it now prints:
    - `Using approximate core-set selection for large job: 56000 selections from 80000 samples.`
  - and then proceeds to:
    - `CelebA: Combined 56000 selected samples with 1000 generated images using Active Learning`
    - `Resumed model from ...`
- Hierarchical patch details:
  - added slice-membership caching keyed by dataset length plus slice name
  - clear the cache at the start of each `select()` call
  - refactored the original within-slice greedy logic into `_greedy_select_within_slice()`
  - for large slices / budgets, hierarchical now first trims to a top-uncertainty candidate pool, then runs the exact greedy uncertainty-plus-diversity selection only inside that reduced pool
  - for smaller jobs, the original exact within-slice greedy path remains in use

### Validation

- [OK] `/people/cs/o/oxx220000/.conda/envs/ad/bin/python -m py_compile utils/hierarchical_al.py utils/active_learning.py`
- [OK] `git diff --check -- utils/hierarchical_al.py utils/active_learning.py .trellis/workspace/oxu/journal-1.md .handoff.md`
- [OK] verified relaunched CoreSet process:
  - `ps -fp 2910596`
- [OK] inspected relaunched CoreSet log:
  - `logs_celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.7_gpu3_rerun_20260416_121045.log`
  - confirmed the approximate-selection line and successful transition into resumed training

### Important Notes

- The hierarchical patch is a first-pass scalability fix focused on the highest-cost inner loop and repeated slice membership work.
- Hierarchical has not yet been rerun after this patch in this session.
- `report.md` was not updated because there are no new stable final metrics yet.

### Status

[OK] **Completed**

---

## Session 65: Mark CelebA hierarchical as deferred

**Date**: 2026-04-16
**Task**: celeba-hierarchical-deferred-note

### Summary

The user decided not to continue the CelebA `hierarchical` experiment for now and asked for a written note. Recorded that decision in the repo continuation documents so future sessions do not keep treating hierarchical as a missing sweep item.

### Main Changes

- Updated:
  - `.handoff.md`
  - `report.md`

### Main Findings

- Current decision:
  - defer CelebA `hierarchical` for now
  - do not treat it as a required pending baseline in the current resume sweep
- Reason for the note:
  - even after the first-pass scalability patch, the GPU `1` rerun
    `logs_celeba/resume_lr1e-5_epoch15_hierarchical_resnet_celeba_1000_ratio0.1_gpu1_rerun_20260416_121310.log`
    still remained stuck at the startup stage

### Validation

- [OK] `git diff --check -- .handoff.md report.md .trellis/workspace/oxu/journal-1.md`

### Important Notes

- This is a planning / sweep-scope note, not a claim that hierarchical is permanently removed.
- The experiment can be revisited later if there is a stronger need to invest in a deeper hierarchical rewrite.

### Status

[OK] **Completed**

---

## Session 63: Launch hierarchical rerun on GPU 1 with ratio 0.1

**Date**: 2026-04-16
**Task**: celeba-hierarchical-gpu1-01-rerun

### Summary

After the first-pass hierarchical scalability patch, the user requested a hierarchical rerun on GPU `1` with ratio `0.1`. Launched the run directly on GPU `1` with the same CelebA resume configuration and confirmed that the worker process is alive and writing the expected startup lines.

### Main Changes

- Launched:
  - `hierarchical`, ratio `0.1`, GPU `1`
  - PID `2919045`
  - log `logs_celeba/resume_lr1e-5_epoch15_hierarchical_resnet_celeba_1000_ratio0.1_gpu1_rerun_20260416_121310.log`
  - pid file `logs_celeba/pids/resume_lr1e-5_epoch15_hierarchical_ratio0.1_gpu1_rerun_20260416_121310.pid`

### Main Findings

- The new worker is alive and currently in normal startup:
  - `Current time: 0416121313`
  - `Selected 1000 filenames for sampling`
  - `Added 1000 generated redhair_brownskin_sademotion images, before training set has 80000 images, now 81000`
- This confirms the rerun launched successfully after the hierarchical code changes.

### Validation

- [OK] verified process:
  - `ps -fp 2919045`
- [OK] inspected startup log:
  - `logs_celeba/resume_lr1e-5_epoch15_hierarchical_resnet_celeba_1000_ratio0.1_gpu1_rerun_20260416_121310.log`

### Important Notes

- The run is still in progress; no final selection or training metrics are available yet.
- `report.md` was not updated because this is an active rerun.

### Status

[OK] **Completed**

---

## Session 64: Remove duplicate GPU 0 BADGE run and launch GPU 2 BADGE queue

**Date**: 2026-04-16
**Task**: celeba-gpu0-dedupe-and-gpu2-badge-queue

### Summary

The user suspected that GPU `0` had duplicate runs and asked to kill one of them, then requested BADGE runs on GPU `2` for ratios `0.5` and `0.7`. Confirmed that GPU `0` indeed had two duplicate BADGE `ratio=0.7` workers alive at the same time, stopped one duplicate, and launched a sequential GPU `2` BADGE controller that runs `0.5` first and then `0.7`.

### Main Changes

- Stopped duplicate GPU `0` BADGE worker:
  - killed PID `2947434`
- Kept active GPU `0` BADGE worker:
  - PID `2945541`
  - strategy `badge`
  - ratio `0.7`
  - GPU `0`
- Launched sequential GPU `2` BADGE controller:
  - controller PID `2950490`
  - controller log `logs_celeba/run_badge_gpu2_ratio0.5_then_0.7_20260416_122256.log`
  - controller pid file `logs_celeba/pids/run_badge_gpu2_ratio0.5_then_0.7_20260416_122256.pid`
- First queued GPU `2` worker log:
  - `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.5_gpu2_20260416_122257.log`

### Main Findings

- GPU `0` duplicate diagnosis:
  - PID `2945541` and PID `2947434` were both:
    - `badge`
    - ratio `0.7`
    - GPU `0`
  - they were duplicate workers for the same configuration
- GPU `2` queue behavior:
  - the controller launched ratio `0.5` first
  - after `0.5` exits, the same controller will launch ratio `0.7`
  - this preserves the user's requested GPU `2` placement without overlapping two training jobs on the same card

### Validation

- [OK] verified after cleanup only the kept GPU `0` worker remains from that duplicate pair:
  - `ps -fp 2945541,2947434,2950490`
- [OK] inspected GPU `2` controller log:
  - `logs_celeba/run_badge_gpu2_ratio0.5_then_0.7_20260416_122256.log`
  - confirmed:
    - `[launch] ratio=0.5 ...`
- [OK] verified the first GPU `2` worker log exists:
  - `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.5_gpu2_20260416_122257.log`

### Important Notes

- The GPU `2` sequential controller is new and still in progress; no final metrics are available yet for `0.5` or `0.7`.
- `report.md` was not updated because these runs are active and incomplete.

### Status

[OK] **Completed**

---

## Session 57: Include base-run test accuracy in resume log messages

**Date**: 2026-04-16
**Task**: resume-print-base-test-acc

### Summary

The user pointed out that resumed runs currently print only:
  - `best val acc`
while the original base-run log already contains the corresponding held-out test accuracy, and asked to record that for future use without rerunning the base model. Updated `alpred.py` so resumed runs now recover the original base-run test accuracy from the existing saved training log and append it to the resume message when available.

### Main Changes

- Updated:
  - `alpred.py`

### Main Findings

- Existing resume message before the patch:
  - `Resumed model from ... (best val acc: 0.914)`
- The base CelebA training log already contains the needed metric:
  - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_20260416_022317.log`
  - `Held-out test accuracy: 0.9140, test loss: 0.3242`
- The active resume path uses:
  - `weights/model_celeba_resnet_base.pth`
  - which is a copied alias, not the original `save/models/...` checkpoint path
- The new helper logic in `alpred.py` therefore:
  - finds the matching original checkpoint file by content when resuming from the shorter base alias
  - finds the log that originally saved that checkpoint
  - extracts the existing overall test accuracy from that log
  - prints it in the resume message if found
- New resume message format:
  - `Resumed model from ... (best val acc: 0.914, base test acc: 0.9140)`

### Validation

- [OK] `/people/cs/o/oxx220000/.conda/envs/ad/bin/python -m py_compile alpred.py`
- [OK] `git diff --check -- alpred.py`
- [OK] verified source log contains the historical test metric:
  - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_20260416_022317.log`

### Important Notes

- Already-running resume jobs will keep their old log wording because they were launched before this patch.
- Newly launched resume jobs will print the recovered base test accuracy when a matching source log can be found.

### Status

[OK] **Completed**

---

## Session 56: Disable verbose CelebA resume launcher output and relaunch `all`

**Date**: 2026-04-16
**Task**: celeba-quiet-all-relaunch

### Summary

The user asked to stop the active CelebA baseline queue, explained that the repeated `Added ... with label ...` lines were too noisy, and requested that `verbose` be set to `false` before restarting `scripts/run_celeba_resume_baselines.sh all`. Verified that the current launcher hard-coded `--verbose True`, killed the active controller and current child jobs, changed the launcher to `--verbose False`, and relaunched the full queue.

### Main Changes

- Updated:
  - `scripts/run_celeba_resume_baselines.sh`
- Stopped prior controller and first-wave workers:
  - old controller PID `2724877`
  - old random-worker PIDs `2724886`, `2724887`, `2724888`, `2724889`
- New controller process:
  - PID `2728053`
- New controller log:
  - `logs_celeba/run_all_resume_baselines_20260416_040048.log`
- New controller pid file:
  - `logs_celeba/pids/run_all_resume_baselines_20260416_040048.pid`

### Main Findings

- The noisy lines came from `alpred.py` under:
  - `if verbose: print(f"Added {image_path} with label {lbl}")`
  - around `alpred.py:1128` and `alpred.py:1155`
- `--verbose` is parsed correctly using:
  - `type=str2bool`
  - so setting `--verbose False` in the launcher disables those prints as intended
- After the relaunch, the worker log no longer prints per-image `Added ... with label ...` lines
- Current startup lines in the relaunched worker log are now concise:
  - `Current time: ...`
  - `Selected 1000 filenames for sampling`
  - `Added 1000 generated ... images`
  - `CelebA: Combined ...`
  - `Resumed model from ...`

### Validation

- [OK] `bash -n scripts/run_celeba_resume_baselines.sh`
- [OK] `git diff --check -- scripts/run_celeba_resume_baselines.sh`
- [OK] verified new controller process:
  - `ps -fp 2728053`
- [OK] inspected new controller log:
  - `logs_celeba/run_all_resume_baselines_20260416_040048.log`
- [OK] inspected worker log after relaunch:
  - `logs_celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.1.log`
  - confirmed the earlier per-image `Added ... with label ...` spam is gone

### Important Notes

- The earlier controller log records the forced stop cleanly:
  - each first-wave random worker exited with status `143`
- The relaunched `all` queue is again running one sequential baseline chain per GPU.

### Status

[OK] **Completed**

---

## Session 55: Launch full CelebA resume baseline queue with `run_celeba_resume_baselines.sh all`

**Date**: 2026-04-16
**Task**: celeba-run-all-baselines

### Summary

The user asked whether `scripts/run_celeba_resume_baselines.sh all` was ready to run and, if so, to start it. Verified that the required base checkpoint exists and there was no existing full-queue controller to reuse, then launched the full detached CelebA baseline queue with the explicit base model path.

### Main Changes

- New controller process:
  - PID `2724877`
- New controller log:
  - `logs_celeba/run_all_resume_baselines_20260416_035907.log`
- New controller pid file:
  - `logs_celeba/pids/run_all_resume_baselines_20260416_035907.pid`

### Main Findings

- The launcher is operational from the current repo state because the earlier blockers had already been fixed:
  - `weights/model_celeba_resnet_base.pth` exists
  - `utils/path_utils.py` compatibility export had already been restored
  - the shared diagnostics crash in `utils/stat.py` had already been fixed
  - the optional AL-vs-random comparison fallback and legacy random-log alias were already in place
- The controller started the four per-GPU sequential queues immediately:
  - GPU `0`, ratio `0.1`
  - GPU `1`, ratio `0.3`
  - GPU `2`, ratio `0.5`
  - GPU `3`, ratio `0.7`
- Initial submitted child jobs from the controller log:
  - `random`, ratio `0.1`, GPU `0`:
    - PID `2724886`
    - log `logs_celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.1.log`
  - `random`, ratio `0.3`, GPU `1`:
    - PID `2724887`
    - log `logs_celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.3.log`
  - `random`, ratio `0.7`, GPU `3`:
    - PID `2724888`
    - log `logs_celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.7.log`
  - `random`, ratio `0.5`, GPU `2`:
    - PID `2724889`
    - log `logs_celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.5.log`

### Validation

- [OK] verified controller process:
  - `ps -fp 2724877`
- [OK] inspected controller log:
  - `logs_celeba/run_all_resume_baselines_20260416_035907.log`
  - confirmed:
    - `Launching per-GPU sequential baseline queues.`
    - all four initial `random` jobs were submitted
    - each GPU queue entered its wait state for the first job

### Important Notes

- This is a long-running controller process; later strategies (`stratified`, `coreset`, `badge`, `hierarchical`) will start sequentially on each GPU after that GPU's current strategy finishes.
- `report.md` was not updated because this step only launched the sweep; there are no stable final sweep results yet.

### Status

[OK] **Completed**

---

## Session 54: Fix shared diagnostics crash in `utils/stat.py` and rerun CelebA stratified gpu2 `0.0001`

**Date**: 2026-04-16
**Task**: celeba-stat-fix-and-rerun-stratified-gpu2

### Summary

The user reported a `TypeError` from `utils/stat.py` after the CelebA `stratified` `sub_ratio=0.0001` / `1` epoch probe on `gpu2` printed its test metric, and asked whether the code path was used by ImageNet. Verified that `alpred.py` calls `MD.detect_prediction_correlation()` for both ImageNet and CelebA, so the shared diagnostics path should remain enabled. Fixed the invalid list indexing in `utils/stat.py`, then reran the same probe and confirmed that it completed cleanly.

### Main Changes

- Updated:
  - `utils/stat.py`
- New rerun process:
  - PID `2721963`
- New rerun log:
  - `logs_celeba/resume_lr1e-5_epoch1_stratified_resnet_celeba_1000_ratio0.0001_gpu2_20260416_035527.log`
- New rerun pid file:
  - `logs_celeba/pids/resume_lr1e-5_epoch1_stratified_ratio0.0001_gpu2_20260416_035527.pid`

### Main Findings

- `detect_prediction_correlation()` is currently used in both dataset branches inside `alpred.py`:
  - `imagenet`
  - `celeba`
- The crash was caused by this invalid Python list access in `utils/stat.py`:
  - `description_valid_idx[20, 30, 40]`
- The fix keeps the shared diagnostics enabled and only replaces the broken preview access with a safe slice:
  - `description_valid_idx[:3]`
- First failed run preserved for reference:
  - `logs_celeba/resume_lr1e-5_epoch1_stratified_resnet_celeba_1000_ratio0.0001_gpu2_20260416_035242.log`
  - it printed the test metric, then failed in diagnostics with the `TypeError`
- Clean rerun outcome:
  - epoch `1/1`: train loss `0.6185`, train acc `0.7569`, val loss `0.4164`, val acc `0.9007`
  - `Model Test ACC: 0.8960, test loss: 0.4180`
  - diagnostics reached:
    - `Model diagnostics complete.`
  - Active Learning versus Random comparison block also completed

### Validation

- [OK] `/people/cs/o/oxx220000/.conda/envs/ad/bin/python -m py_compile utils/stat.py`
- [OK] `git diff --check -- utils/stat.py`
- [OK] inspected clean rerun log:
  - `logs_celeba/resume_lr1e-5_epoch1_stratified_resnet_celeba_1000_ratio0.0001_gpu2_20260416_035527.log`
  - confirmed:
    - no traceback
    - `Model diagnostics complete.`
    - final comparison sections printed through the end of the run

### Important Notes

- The underlying train/validation/test metrics of the failed first run and the clean rerun matched:
  - the fix only changed the post-evaluation diagnostics print path
- `report.md` was not updated because this was a one-off probe plus bug fix, not a report-level stable sweep summary.

### Status

[OK] **Completed**

---

## Session 53: Launch CelebA stratified probe on gpu2 with `sub_ratio=0.0001`

**Date**: 2026-04-16
**Task**: celeba-stratified-gpu2-0001

### Summary

The user asked for an `alpred.py` CelebA `stratified` resume run on `gpu2` with `sub_ratio=0.0001` and `1` epoch. Verified the current launcher pattern and `alpred.py` GPU resolution logic, then launched the run directly on `gpu2` and confirmed that it is alive and writing normal startup output.

### Main Changes

- Process launch:
  - PID `2719983`
- New log:
  - `logs_celeba/resume_lr1e-5_epoch1_stratified_resnet_celeba_1000_ratio0.0001_gpu2_20260416_035242.log`
- New pid file:
  - `logs_celeba/pids/resume_lr1e-5_epoch1_stratified_ratio0.0001_gpu2_20260416_035242.pid`

### Main Findings

- The repo sweep launcher `scripts/run_celeba_resume_baselines.sh` only hard-codes:
  - ratios `0.1`, `0.3`, `0.5`, `0.7`
  - GPUs `0`, `1`, `2`, `3`
- For this user request, the run needed a direct manual launch instead of the fixed sweep script.
- `alpred.py` now correctly maps the requested GPU through `_resolve_cuda_device_index(...)`, so using:
  - `CUDA_VISIBLE_DEVICES=2`
  - `--gpu 2`
  is valid in the current code path.

### Validation

- [OK] verified launched process:
  - `ps -fp 2719983`
- [OK] inspected startup log:
  - `logs_celeba/resume_lr1e-5_epoch1_stratified_resnet_celeba_1000_ratio0.0001_gpu2_20260416_035242.log`
  - current startup lines show:
    - current time `0416035245`
    - selected `1000` generated filenames
    - normal CelebA generated-set loading before Active Learning selection/training

### Important Notes

- This run was launched with the same resume-from-base CelebA settings used for the other recent `epoch=1` probes:
  - `resume_model=weights/model_celeba_resnet_base.pth`
  - `lr=1e-5`
  - `num_added_img=1000`
- The run is still active at the time of this journal entry, so there are no final metrics yet.

### Status

[OK] **Completed**

---

## Session 52: Standardize runtime test metric wording to `Model Test ACC`

**Date**: 2026-04-16
**Task**: standardize-test-metric-label

### Summary

The user asked whether CelebA and ImageNet10 were using different test-metric wording, and requested that both use `Model Test ACC`. Verified that the active runtime path in `alpred.py` currently printed `Held-out test accuracy` for both datasets, then changed the shared runtime output and parser helpers so future runs consistently use `Model Test ACC`.

### Main Changes

- Updated:
  - `alpred.py`
  - `utils/parse_sweep.py`
  - `utils/update_intermediate_report.py`
  - `utils/compare_acc.py`
  - `utils/training.py`

### Main Findings

- Current runtime behavior before the patch:
  - `alpred.py` printed:
    - `Held-out test accuracy: ...`
  - for both CelebA and ImageNet10
- Historical helper scripts in the repo were mixed:
  - some expected:
    - `Held-out test accuracy: ...`
  - others already expected:
    - `Model Test ACC: ...`
- New standardized runtime behavior:
  - `alpred.py` now prints:
    - `Model Test ACC: {test_acc:.4f}, test loss: {test_loss:.4f}`
- Parser compatibility behavior:
  - helper regexes now accept all of:
    - `Model Test ACC`
    - `Held-out test accuracy`
    - `Model Validation ACC`
  - this keeps old logs parseable while future logs use the standardized label

### Validation

- [OK] `/people/cs/o/oxx220000/.conda/envs/ad/bin/python -m py_compile alpred.py utils/parse_sweep.py utils/update_intermediate_report.py utils/compare_acc.py utils/training.py`
- [OK] `git diff --check -- alpred.py utils/parse_sweep.py utils/update_intermediate_report.py utils/compare_acc.py utils/training.py`

### Important Notes

- Already-running jobs will still emit the old wording because they started before this code change.
- Newly launched runs for both CelebA and ImageNet10 will use:
  - `Model Test ACC: ...`

### Status

[OK] **Completed**

---

## Session 51: Suppress torchvision pretrained warnings and stop printing `Model Validation ACC`

**Date**: 2026-04-16
**Task**: celeba-log-cleanup-heldout-focused

### Summary

The user asked to remove the noisy torchvision pretrained/weights deprecation warnings from the logs and to stop reporting `Model Validation ACC:` because the desired headline metric is the held-out test accuracy. Updated the shared import warning filters and removed the diagnostics print for `Model Validation ACC`.

### Main Changes

- Updated:
  - `utils/imports.py`
  - `utils/stat.py`
  - `utils/idx.py`

### Main Findings

- The two repeated warnings were emitted from `torchvision.models._utils` when ResNet was instantiated using the older `pretrained`/`weights` calling path.
- The warnings are now suppressed centrally in `utils/imports.py` using `warnings.filterwarnings(...)`, so future runs that import the shared module do not spam those lines.
- `Model Validation ACC:` was not the main metric the user wanted in these resume logs.
- The print came from the diagnostics helper itself, not from the training/evaluation path.
- Removed the `Model Validation ACC:` print in:
  - `utils/stat.py`
  - `utils/idx.py`
- The held-out evaluation line remains unchanged:
  - `Held-out test accuracy: ...`

### Validation

- [OK] `/people/cs/o/oxx220000/.conda/envs/ad/bin/python -m py_compile utils/imports.py utils/stat.py utils/idx.py`
- [OK] `git diff --check -- utils/imports.py utils/stat.py utils/idx.py`

### Important Notes

- These changes only affect future process launches.
- Already-running jobs or already-written logs will still contain the older warning lines and the older `Model Validation ACC:` print if they started before this patch.

### Status

[OK] **Completed**

---

## Session 50: Launch CelebA hierarchical probe on gpu3 with `sub_ratio=0.0001`

**Date**: 2026-04-16
**Task**: celeba-hierarchical-gpu3-0001

### Summary

The user asked for a new CelebA `hierarchical` resume run on `gpu3` with `sub_ratio=0.0001` and `1` epoch. Launched it directly on `gpu3` without a prior GPU-availability check, and verified that the process is alive and writing to `logs_celeba/`.

### Main Changes

- Process launch:
  - PID `2715276`
- New log:
  - `logs_celeba/resume_lr1e-5_epoch1_hierarchical_resnet_celeba_1000_ratio0.0001_gpu3_20260416_034450.log`
- New pid file:
  - `logs_celeba/pids/resume_lr1e-5_epoch1_hierarchical_ratio0.0001_gpu3_20260416_034450.pid`

### Validation

- [OK] verified launched process:
  - `ps -fp 2715276`
- [OK] inspected startup log:
  - `logs_celeba/resume_lr1e-5_epoch1_hierarchical_resnet_celeba_1000_ratio0.0001_gpu3_20260416_034450.log`
  - current startup lines show:
    - current time `0416034452`
    - selected `1000` generated filenames
    - CelebA generated set added before AL/training setup

### Important Notes

- This run was launched in the standardized repo-log style after the earlier mistake with the foreground-only hierarchical probe:
  - stdout/stderr now go to `logs_celeba/...`
- The run is still active at the time of this journal entry, so there are no final metrics yet.

### Status

[OK] **Completed**

---

## Session 49: Make AL-vs-random comparison logs optional in `alpred.py`

**Date**: 2026-04-16
**Task**: celeba-optional-compare-log

### Summary

The user asked that missing random-reference logs must never abort a run. Updated the Active Learning versus Random comparison block in `alpred.py` so if either comparison log exists it is used, and if both are missing the code only prints a skip message and continues to the end of the run.

### Main Changes

- Updated:
  - `alpred.py`

### Main Findings

- The old code already accepted either comparison-log path:
  - `logs/AL_resnet_celeba_random_1000.log`
  - `logs/AL_random_resnet_celeba_1000_ratio<sub_ratio>.log`
- The actual failure mode was only when **both** were missing:
  - it raised:
    - `FileNotFoundError`
  - which aborted an otherwise completed training/evaluation run late in execution
- The new behavior is:
  - if `log_path2` exists, use it
  - else if `log_path1` exists, use it
  - else:
    - print a skip message
    - do not raise
    - continue normally

### Validation

- [OK] `/people/cs/o/oxx220000/.conda/envs/ad/bin/python -m py_compile alpred.py`
- [OK] inspected patched block:
  - `alpred.py:2118-2129`

### Important Notes

- This makes the comparison block operationally optional, which matches the user's requirement that missing reference logs must not prevent a run from finishing.
- Existing already-finished processes will not retroactively pick up this change; only future runs use the new behavior.

### Status

[OK] **Completed**

---

## Session 48: Kill stray hierarchical probe, add CelebA random-log alias, and launch gpu3 coreset `0.0001`

**Date**: 2026-04-16
**Task**: celeba-log-alias-and-gpu3-coreset-0001

### Summary

The user asked to stop the old CelebA `hierarchical` run, fix the missing comparison-log path by copying the finished base log into the expected legacy filename, and then launch a new CelebA `coreset` resume probe on `gpu3` with `sub_ratio=0.0001` and `1` epoch. Completed all three.

### Main Changes

- Filesystem updates:
  - created:
    - `logs/AL_resnet_celeba_random_1000.log`
- Process changes:
  - stopped stray hierarchical run:
    - PID `2703500`
  - launched new gpu3 coreset run:
    - PID `2712644`
    - log:
      - `logs_celeba/resume_lr1e-5_epoch1_coreset_resnet_celeba_1000_ratio0.0001_gpu3_20260416_034042.log`
    - pid file:
      - `logs_celeba/pids/resume_lr1e-5_epoch1_coreset_ratio0.0001_gpu3_20260416_034042.pid`

### Main Findings

- The active-learning comparison section in `alpred.py` still looks for CelebA random-reference logs under `logs/`, not `logs_celeba/`.
- For the current CelebA AL runs, the missing lookup was:
  - `logs/AL_resnet_celeba_random_1000.log`
  - fallback:
    - `logs/AL_random_resnet_celeba_1000_ratio<sub_ratio>.log`
- The user specified that:
  - `logs/AL_resnet_celeba_random_1000.log`
  - should be a renamed copy of the finished base-model log
- Copied the finished base log:
  - source:
    - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_20260416_022317.log`
  - destination:
    - `logs/AL_resnet_celeba_random_1000.log`
- This avoids the previous end-of-run failure:
  - `FileNotFoundError: No log file found: logs/AL_resnet_celeba_random_1000.log or logs/AL_random_resnet_celeba_1000_ratio0.001.log`

### Validation

- [OK] stopped hierarchical PID:
  - `kill 2703500`
- [OK] confirmed copied legacy compare log exists:
  - `ls -l logs/AL_resnet_celeba_random_1000.log`
- [OK] launched requested gpu3 coreset run:
  - `ps -fp 2712644`
- [OK] confirmed startup log under repo log dir:
  - `logs_celeba/resume_lr1e-5_epoch1_coreset_resnet_celeba_1000_ratio0.0001_gpu3_20260416_034042.log`
- [OK] confirmed gpu3 occupancy:
  - `GPU-e3ce52a2-3737-fae7-a3bd-9adfee05cbc5, 2712644, /people/cs/o/oxx220000/.conda/envs/ad/bin/python, 1250 MiB`

### Important Notes

- The new run was launched directly on `gpu3` without a prior availability check, following the repo-local GPU assignment rule the user requested.
- The new run is a `coreset` probe because the user request followed the currently active CelebA `coreset` troubleshooting thread and did not specify a different strategy.
- At the time of this journal entry, the `gpu3` `coreset` `0.0001` run is still active, so no final metrics are recorded yet.

### Status

[OK] **Completed**

---

## Session 47: Fix nonzero-GPU CUDA mapping for `alpred.py` and relaunch CelebA AL probes

**Date**: 2026-04-16
**Task**: celeba-nonzero-gpu-al-launch-fix

### Summary

After the user asked for a 1-epoch CelebA `coreset` resume run on `gpu1`, the first launch failed with `RuntimeError: CUDA error: invalid device ordinal`. Traced this to `alpred.py` using `torch.device(f"cuda:{args.gpu}")` directly even when the process was launched with `CUDA_VISIBLE_DEVICES=<physical_gpu_id>`. Added runtime CUDA-index normalization so masked single-GPU launches on `gpu1/2/3` work correctly, relaunched the requested `gpu1` `coreset` probe, then replaced its `sub_ratio=0.1` run with the user's requested `sub_ratio=0.001` run.

### Main Changes

- Updated:
  - `alpred.py`
  - `AGENTS.md`

### Main Findings

- Root cause of the failed `gpu1` launch:
  - the process was started with:
    - `CUDA_VISIBLE_DEVICES=1`
    - `--gpu 1`
  - but inside that process only one CUDA device is visible, with local index `0`
  - several `alpred.py` call sites still used:
    - `torch.device(f"cuda:{args.gpu}")`
  - this makes nonzero physical GPU launches fail in masked environments, especially during AL selection and global model-device setup
- Added two helpers in `alpred.py`:
  - `_resolve_cuda_device_index(requested_gpu)`
  - `_resolve_runtime_device(requested_gpu)`
- Replaced all relevant runtime device uses so they now:
  - map physical GPU ids through `CUDA_VISIBLE_DEVICES` when present
  - fall back safely to local CUDA index `0` when exactly one masked device is visible
  - use the normalized index for:
    - AL selection device setup
    - LLM-selection device setup
    - global `torch.cuda.set_device(...)`
    - main model `device`
- Added a repo-local instruction in `AGENTS.md`:
  - if the user explicitly tells the agent to run on a specific GPU, do not perform a separate GPU-availability check first

### Commands and Artifacts

- Failed first `gpu1` coreset attempt:
  - log:
    - `logs_celeba/resume_lr1e-5_epoch1_coreset_resnet_celeba_1000_ratio0.1_gpu1_20260416_033506.log`
  - failure:
    - `RuntimeError: CUDA error: invalid device ordinal`
- Replacement `gpu1` coreset launch with the CUDA fix applied:
  - first relaunched `sub_ratio=0.1` run:
    - PID `2709238`
    - log:
      - `logs_celeba/resume_lr1e-5_epoch1_coreset_resnet_celeba_1000_ratio0.1_gpu1_20260416_033626.log`
  - user then requested:
    - `sub_ratio=0.001`
  - stopped PID:
    - `2709238`
  - active replacement run:
    - PID `2710039`
    - log:
      - `logs_celeba/resume_lr1e-5_epoch1_coreset_resnet_celeba_1000_ratio0.001_gpu1_20260416_033705.log`
- Still-active earlier hierarchical probe from the previous session step:
  - PID `2703500`
  - command family:
    - CelebA `hierarchical`
    - `gpu0`
    - `sub_ratio=0.1`
    - `num_epochs=1`

### Validation

- [OK] confirmed failed first `gpu1` coreset log shows the original bug:
  - `invalid device ordinal`
- [OK] `rg -n --fixed-strings 'cuda:{args.gpu}' alpred.py`
  - returns no remaining literal matches after the patch
- [OK] `/people/cs/o/oxx220000/.conda/envs/ad/bin/python -m py_compile alpred.py`
- [OK] verified replacement `gpu1` run is alive:
  - `ps -fp 2710039`
- [OK] verified startup log for the replacement run:
  - `logs_celeba/resume_lr1e-5_epoch1_coreset_resnet_celeba_1000_ratio0.001_gpu1_20260416_033705.log`
- [OK] verified GPU occupancy:
  - `gpu0` still held by hierarchical PID `2703500`
  - `gpu1` held by coreset PID `2710039`

### Important Notes

- This CUDA-index normalization is operationally important for:
  - `scripts/run_celeba_resume_baselines.sh`
  - any future direct launches using:
    - `CUDA_VISIBLE_DEVICES=<physical_id>`
    - `--gpu <physical_id>`
- Without this fix, nonzero GPU launches could fail during AL setup even though the GPU itself was fine.
- The current `gpu1` coreset `sub_ratio=0.001` run is still active at the time of this journal entry, so there are no final metrics yet.

### Status

[OK] **Completed**

---

## Session 46: Fix CelebA resume diagnostics mismatch and verify 1-epoch random probe

**Date**: 2026-04-16
**Task**: celeba-random-probe-and-diagnostics-fix

### Summary

The user asked to take the finished CelebA base checkpoint, copy it to a shorter stable name for `--resume_model`, and try a `gpu0` CelebA resume run for `sub_ratio=0.1` with the `random` baseline. After the user clarified that a 1-epoch probe was enough, copied the checkpoint to `weights/model_celeba_resnet_base.pth`, found and fixed two runtime blockers, and verified that the 1-epoch CelebA random resume run now completes diagnostics cleanly.

### Main Changes

- Updated:
  - `utils/path_utils.py`
  - `alpred.py`
- Added base-model copy:
  - `weights/model_celeba_resnet_base.pth`

### Main Findings

- The first launch attempt failed before training because `alpred.py` still imported `LEGACY_SAVE_ROOT` even though the save-root migration removed that symbol from `utils/path_utils.py`.
- After restoring a compatibility constant, the CelebA random probe ran through training and evaluation but crashed in diagnostics with:
  - `IndexError: list index out of range`
  - source:
    - `utils/stat.py:init_data_static()`
- This was **not** caused by a missing val/test split.
- The actual root cause was a diagnostics alignment bug in the CelebA path:
  - saved predictions for subsampled resume runs include:
    - padded original-train predictions
    - validation predictions
    - held-out test predictions
    - generated-image predictions
    - unlabeled predictions
  - but CelebA attribute metadata only covers the original images, not generated samples
  - the old diagnostics path used `combined_labels` and a shorter split list, so `ModelDiagnose` eventually indexed past the end of `split`
- The diagnostics fix now:
  - builds CelebA diagnostics labels from original-image splits only:
    - `train_labels`
    - `valid_labels`
    - `test_labels`
    - `unlabeled_labels`
  - strips the generated-image prediction segment before instantiating `ModelDiagnose`
  - skips diagnostics with a warning if a future predictions length does not match either the original-only or generated-inclusive expectation

### Validation

- [OK] confirmed short base-model copy:
  - `weights/model_celeba_resnet_base.pth`
- [OK] first failing detached probe log:
  - `logs_celeba/resume_lr1e-5_epoch1_random_resnet_celeba_1000_ratio0.1_20260416_031858.log`
  - showed:
    - training finished
    - held-out test accuracy `0.9095`
    - diagnostics crash from split-length mismatch
- [OK] reran the same CelebA probe after the fix in the working conda environment:
  - command:
    - `env MPLCONFIGDIR=/tmp/matplotlib-safefix CUDA_VISIBLE_DEVICES=0 /people/cs/o/oxx220000/.conda/envs/ad/bin/python -u alpred.py --dataset celeba --model resnet --attributes redhair_brownskin_sademotion --gen_image_dir data/redhair_brownskin_sademotion --num_added_img 1000 --num_epochs 1 --resume_model weights/model_celeba_resnet_base.pth --freeze_backbone False --lr 1e-5 --batch_size 64 --workers 4 --seed 42 --gpu 0 --save_pred_model best --save_preds True --save_model False --verbose False --sub_ratio 0.1 --use_active_learning False`
- [OK] verified clean post-fix outcome from the rerun:
  - best-val model resumed from:
    - `weights/model_celeba_resnet_base.pth`
  - epoch `1/1`:
    - train loss `0.2605`
    - train acc `0.9608`
    - val loss `0.3448`
    - val acc `0.9101`
  - held-out test:
    - accuracy `0.9095`
    - loss `0.3480`
  - diagnostics:
    - `Model Validation ACC: 0.9101`
    - completed through:
      - `detect_failure_by_label`
      - `detect_prediction_correlation`
      - `detect_failure_prediction_correlation`
    - ended with:
      - `Model diagnostics complete.`

### Important Notes

- Because `scripts/run_celeba_resume_baselines.sh` waits on each Python exit code inside the per-GPU `all` pipelines, this diagnostics fix matters operationally:
  - without it, the first CelebA job on each GPU would finish training but still exit nonzero
  - with the fix, the 1-epoch random probe now exits cleanly end-to-end
- `report.md` was not updated because this was a short operational verification plus runtime-fix session, not a stable final experiment sweep.

### Status

[OK] **Completed**

---

## Session 45: Complete `outputs/save` to `save` artifact-root migration

**Date**: 2026-04-16
**Task**: save-root-migration-completion

### Summary

The user asked whether the remaining `outputs/save` tree could now be moved into `save/`, and requested a repo-wide check for any code that still used or wrote to `outputs/save`. Verified the current state, removed the last code/config dependency on the legacy path, merged the remaining legacy artifacts into `save/`, and removed the stale `outputs/save/` tree.

### Main Changes

- Updated:
  - `.gitignore`
  - `utils/path_utils.py`
- Filesystem migration:
  - moved `319` files that still existed only under `outputs/save/` into `save/`
  - removed `686` duplicate legacy files after confirming they matched the `save/` copies
  - removed `1` older legacy file after confirming the `save/` copy was newer:
    - `celeba_split_meta.json`
  - removed the now-empty `outputs/save/` directory tree

### Main Findings

- Before this cleanup:
  - executable-code search for literal `outputs/save` references returned only one stale config match:
    - `.gitignore`
  - `utils/path_utils.py` still had a legacy read fallback that could resolve artifacts from `outputs/save/`
  - `save/` was already partially populated, so this was a merge rather than a simple rename
- Overlap analysis before the migration:
  - `319` files existed only in `outputs/save/`
  - `687` relative paths existed in both trees
  - only `1` overlapping file actually differed in content:
    - `celeba_split_meta.json`
  - the `save/celeba_split_meta.json` copy was newer than the legacy one, so the legacy copy was discarded
- After the cleanup:
  - `utils/path_utils.py` no longer falls back to `outputs/save/`
  - passing either `save/...` or `outputs/save/...` into the resolver now canonicalizes to `save/...`
  - the repo no longer contains an `outputs/save/` directory

### Validation

- [OK] executable-code search for legacy path:
  - `rg -n --hidden --glob '!**/*.md' --glob '!**/*.html' --glob '!**/*.log' --glob '!**/*.json' --glob '!**/*.csv' --glob '!**/*.npy' --glob '!**/*.txt' --glob '!**/*.ipynb' "outputs/save" .`
  - returned no matches
- [OK] filesystem check:
  - `find outputs -maxdepth 3 \( -type d -o -type l \) | sort`
  - no longer shows `outputs/save`
- [OK] path-resolution probe:
  - `python3 - <<'PY' ...`
  - `resolve_artifact_path('save/attribute10.json')`
  - `resolve_artifact_path('outputs/save/attribute10.json')`
  - `resolve_existing_artifact('save/attribute10.json')`
  - all resolve to:
    - `/people/cs/o/oxx220000/safefix/save/attribute10.json`
- [OK] `git diff --check`

### Important Notes

- `report.md` was not updated because this session changed artifact-path behavior and filesystem layout, not a stable human-facing experiment result.
- The repo now expects `save/` to be the single canonical artifact root.
- Historical notes and old report text may still mention `outputs/save/` when describing past states, but active code/config resolution should not rely on that path anymore.

### Status

[OK] **Completed**

---

## Session 44: Confirm CelebA base completion and add stable result to `report.md`

**Date**: 2026-04-16
**Task**: celeba-base-result-report-update

### Summary

The user asked whether the earlier CelebA base run had finished, where its real log lives, and requested that the stable result be written into `report.md`. Verified that the original GPU `0` PID is no longer alive, located the complete training log, confirmed the saved checkpoint and prediction artifact, and added a new report section summarizing the finished CelebA ResNet base run.

### Main Findings

- The short startup log:
  - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_gpu0.log`
  - only contains the initial launch lines and is not the full training record
- The complete finished log is:
  - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_20260416_022317.log`
- The completed run used:
  - dataset: `CelebA`
  - model: `ResNet`
  - attributes: `redhair_brownskin_sademotion`
  - generated images: `0`
  - initialization: random weights
  - epochs: `20`
- Extracted stable metrics:
  - best val acc: `0.9140`
  - best-val epoch: `12`
  - final val acc: `0.9066`
  - held-out test acc: `0.9140`
  - test loss: `0.3242`
  - train time: `1939.07 s`
  - end-to-end time: `1944.28 s`
- Saved artifacts confirmed:
  - checkpoint:
    - `save/models/model_celeba_resnet_redhair_brownskin_sademotion_0imgs_0416022326.pth`
  - predictions:
    - `save/pred/predictions_celeba_resnet_redhair_brownskin_sademotion_0imgs_0416022326.npy`
- The run failed only after the main outputs were already produced:
  - diagnostics crashed because `save/attribute10.json` was missing
  - this happened after checkpoint save, test evaluation, and prediction dump

### Files Updated

- `report.md`

### Validation

- [OK] `ps -fp 2646523`
  - original run PID is no longer alive
- [OK] inspected complete log:
  - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_20260416_022317.log`
- [OK] confirmed checkpoint timestamp:
  - `save/models/model_celeba_resnet_redhair_brownskin_sademotion_0imgs_0416022326.pth`
- [OK] confirmed prediction artifact exists:
  - `save/pred/predictions_celeba_resnet_redhair_brownskin_sademotion_0imgs_0416022326.npy`

### Important Notes

- This finished base checkpoint is the intended `--resume_model` candidate for the upcoming CelebA AL follow-up runs.
- `report.md` now records the complete log path explicitly so later sessions do not mistake the short `gpu0` launch log for the real training log.

### Status

[OK] **Completed**

---

## Session 43: Change CelebA launcher `all` mode to per-GPU sequential queues

**Date**: 2026-04-16
**Task**: celeba-launcher-all-mode-per-gpu

### Summary

After reviewing the initial `all` implementation for `scripts/run_celeba_resume_baselines.sh`, the user clarified that each GPU should be treated independently. Updated `all` mode so it no longer waits for all four ratio jobs of one baseline to finish before moving on. Instead, each GPU now runs its own baseline queue sequentially for its assigned ratio.

### Main Changes

- Updated:
  - `scripts/run_celeba_resume_baselines.sh`
- Added per-GPU sequence helpers so `all` mode now behaves as:
  - GPU `0`, ratio `0.1`:
    - `random -> stratified -> coreset -> badge -> hierarchical`
  - GPU `1`, ratio `0.3`:
    - `random -> stratified -> coreset -> badge -> hierarchical`
  - GPU `2`, ratio `0.5`:
    - `random -> stratified -> coreset -> badge -> hierarchical`
  - GPU `3`, ratio `0.7`:
    - `random -> stratified -> coreset -> badge -> hierarchical`
- Updated script help text to state explicitly that `all` mode is per-GPU sequential.

### Validation

- [OK] `bash -n scripts/run_celeba_resume_baselines.sh`
- [OK] `./scripts/run_celeba_resume_baselines.sh all --base-model weights/model_imagenet_resnet_base.pth --dry-run`
- [OK] `git diff --check`

### Important Notes

- `all` mode now launches four controller shells in parallel, one per `(gpu, ratio)` pair.
- Within each controller, the next baseline starts only after the previous baseline on that same GPU exits.
- The dry-run output is still printed sequentially for readability; the real `all` run uses background controller processes.
- `report.md` was not updated because this change only affects launch orchestration.

### Status

[OK] **Completed**

---

## Session 42: Add CelebA resume baseline launcher and fix CelebA hierarchical / resume LR handling

**Date**: 2026-04-16
**Task**: celeba-resume-baselines-script-and-fixes

### Summary

The user asked for a new executable shell script, not `draft.sh`, that prepares the CelebA resume-from-base baselines for `random`, `stratified`, `coreset`, `badge`, and `hierarchical`, with ratio `0.1/0.3/0.5/0.7` pinned to GPUs `0/1/2/3`. The user also asked to complete two follow-up code tasks: fix the broken CelebA hierarchical mapping and make resumed CelebA ResNet runs automatically drop to `lr=1e-5`. Implemented all three.

### Main Changes

- Added a new executable launcher script:
  - `scripts/run_celeba_resume_baselines.sh`
- Updated `alpred.py`:
  - resumed CelebA ResNet runs now automatically switch from the default `lr=1e-4` to `lr=1e-5`
- Updated `utils/hierarchical_al.py`:
  - CelebA hierarchical groups now use real attribute keys that exist in `data/celebA/attribute.json`
  - generic attribute-JSON conversion now supports the repo's common `attribute -> {"data": [...]}` layout
  - hierarchical selection now loads CelebA samplewise metadata instead of treating the raw JSON as if it were already sample-index keyed

### Main Findings

- The old CelebA hierarchical implementation was effectively misaligned with the current metadata:
  - old group keys included:
    - `hair`
    - `facial`
    - `skin`
  - but the actual CelebA attribute file uses keys such as:
    - `hair color`
    - `emotion`
    - `skin`
    - `beard`
- After the fix, instantiating `HierarchicalActiveStrategy(dataset='celeba')` reports:
  - group keys:
    - `hair color`
    - `emotion`
    - `skin`
    - `beard`
  - group sizes:
    - `hair color: 7`
    - `emotion: 9`
    - `skin: 3`
    - `beard: 2`
  - samplewise metadata coverage:
    - `180000` samples
- The new shell launcher is intentionally strategy-at-a-time:
  - one invocation launches one baseline across four GPUs
  - this avoids accidentally dumping all `5 x 4 = 20` jobs onto the same four GPUs at once

### Commands and Artifacts

- New script usage:
  - `./scripts/run_celeba_resume_baselines.sh random`
  - `./scripts/run_celeba_resume_baselines.sh stratified`
  - `./scripts/run_celeba_resume_baselines.sh coreset`
  - `./scripts/run_celeba_resume_baselines.sh badge`
  - `./scripts/run_celeba_resume_baselines.sh hierarchical`
- The script auto-discovers the latest matching base model under:
  - `save/models/model_celeba_resnet_redhair_brownskin_sademotion_0imgs_*.pth`
  - falling back to:
    - `outputs/save/models/model_celeba_resnet_redhair_brownskin_sademotion_0imgs_*.pth`
- Strategy-specific parameters baked into the launcher:
  - `random`
    - `--use_active_learning False`
  - `stratified`
    - `--al_strategy stratified`
  - `coreset`
    - `--al_strategy coreset --al_embedding_dim 512 --al_initial_labeled 1000`
  - `badge`
    - `--al_strategy badge --al_embedding_dim 512`
  - `hierarchical`
    - `--al_strategy hierarchical --al_embedding_dim 512`
    - `--al_round1_ratio 0.8`
    - `--al_alpha 0.5 --al_beta 0.35 --al_gamma 0.15`
    - `--al_lambda_div 0.3`
    - `--al_min_slice_quota 20 --al_max_slice_quota 500`

### Validation

- [OK] `python3 -m py_compile alpred.py utils/hierarchical_al.py`
- [OK] `bash -n scripts/run_celeba_resume_baselines.sh`
- [OK] `git diff --check`
- [OK] dry-run launcher probe:
  - `./scripts/run_celeba_resume_baselines.sh badge --base-model weights/model_imagenet_resnet_base.pth --dry-run`
- [OK] CelebA hierarchical metadata probe:
  - confirmed real keys and samplewise conversion from `data/celebA/attribute.json`

### Important Notes

- The launcher hard-codes the intended CelebA setup:
  - `attributes=redhair_brownskin_sademotion`
  - `num_added_img=1000`
  - `num_epochs=15`
  - `freeze_backbone=False`
  - `lr=1e-5`
- The launcher now also supports:
  - `./scripts/run_celeba_resume_baselines.sh all`
- `all` mode is sequential, not parallel across strategies:
  - launch order:
    - `random`
    - `stratified`
    - `coreset`
    - `badge`
    - `hierarchical`
  - each strategy launches its four GPU-specific ratio jobs first
  - then the script waits for all four launched PIDs to finish before starting the next strategy
- The script is ready to use after the base model checkpoint exists; it does not itself wait for the base run to finish.
- `report.md` was not updated because this session changed launch tooling and selector correctness, not stable experiment results.

### Status

[OK] **Completed**

---

## Session 41: Restore `save/pred` compatibility and relaunch CelebA ResNet base run

**Date**: 2026-04-16
**Task**: celeba-base-relaunch-and-save-pred-compat

### Summary

The user asked whether an earlier CelebA ResNet checkpoint still exists for `--resume_model`, and if not, to relaunch a fresh CelebA ResNet run on GPU `0`. Investigation showed that the repo still contains many saved CelebA prediction arrays, but no reusable CelebA ResNet `.pth` checkpoint. While checking the ongoing ImageNet BADGE resume runs, the ratio-`0.1` job had already crashed after training finished because `alpred.py` tried to save predictions into `save/pred/`, but the current repo state only kept the predictions under `outputs/save/pred/`. Restored backward compatibility by recreating `save/pred` as a symlink to `outputs/save/pred` and hardened `alpred.py` so it now creates `save/pred` before writing predictions. Then launched a new from-scratch CelebA ResNet baseline on GPU `0` with `--save_model True` so a reusable checkpoint will exist after completion.

### Main Findings

- No reusable CelebA ResNet checkpoint is currently present in the repo:
  - no `model_celeba_resnet*.pth` was found under the workspace
- Existing CelebA prediction artifacts are present under:
  - `outputs/save/pred/`
  - examples include:
    - `outputs/save/pred/predictions_celeba_resnet_redhair_brownskin_0imgs_1112234817.npy`
    - `outputs/save/pred/predictions_celeba_resnet_redhair_brownskin_sademotion_1000imgs_0730193801.npy`
- The ImageNet BADGE resume run for ratio `0.1` already completed training and test evaluation, then failed only at prediction saving:
  - log: `logs_imagenet/resume_lr5_epoch15_badge_resnet_imagenet_1000_ratio0.1.log`
  - final printed held-out test accuracy before the crash:
    - `0.7323`
  - crash cause:
    - missing directory `save/pred/`
- The other BADGE ImageNet resume jobs were still alive after the compatibility fix:
  - ratio `0.3`: PID `2634921`
  - ratio `0.5`: PID `2634930`
  - ratio `0.7`: PID `2634936`

### Main Changes

- Restored legacy prediction output path compatibility:
  - created symlink:
    - `save/pred -> ../outputs/save/pred`
- Hardened prediction saving in:
  - `alpred.py`
  - `save_predictions()` now ensures `save/pred` exists before calling `np.save`

### Commands and Artifacts

- Confirmed missing CelebA checkpoints and existing predictions with targeted `find` / `rg` queries.
- Relaunched fresh CelebA ResNet base run on GPU `0`:
  - command:
    - `CUDA_VISIBLE_DEVICES=0 /people/cs/o/oxx220000/.conda/envs/ad/bin/python -u alpred.py --dataset celeba --model resnet --attributes redhair_brownskin_sademotion --gen_image_dir data/redhair_brownskin_sademotion --num_added_img 0 --use_active_learning False --sub_ratio 1.0 --num_epochs 20 --gpu 0 --save_model True`
  - live Python PID:
    - `2646523`
  - launch PID file:
    - `logs/pids/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_gpu0.pid`
  - log:
    - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_gpu0.log`

### Validation

- [OK] `python3 -m py_compile alpred.py`
- [OK] `git diff --check`
- [OK] `readlink -f save/pred`
  - resolves to `outputs/save/pred`
- [OK] `ps -fp 2646523 2634921 2634930 2634936`

### Important Notes

- The ratio-`0.1` ImageNet BADGE run is already dead, so restoring `save/pred` cannot retroactively generate its missing prediction `.npy`. If that file is needed, that ratio must be rerun.
- The new CelebA base run was launched with `--save_model True`; once it finishes, its checkpoint path should follow:
  - `save/models/model_celeba_resnet_redhair_brownskin_sademotion_0imgs_<timestamp>.pth`
- `report.md` was not updated because neither the relaunched CelebA baseline nor the remaining BADGE runs have stable final summarized results yet.

### Status

[OK] **In Progress**

---

## Session 42: Stop new artifacts from flowing into `outputs/save`

**Date**: 2026-04-16
**Task**: stop-outputs-save-writes

### Summary

The user asked whether the repo still had executable code writing into `outputs/save`, and requested that this be fixed to prevent future outputs from landing there. Searched the repo excluding docs/logs/artifacts, found that the remaining practical problem was not broad hard-coded writes but path aliasing: `save/pred` was a symlink to `outputs/save/pred`, so any code writing to `save/pred/...` still created new files under `outputs/save/pred`. Also found one remaining executable script that explicitly searched `outputs/save/models`.

### Main Changes

- Removed the last explicit executable-script reference to `outputs/save`:
  - updated `scripts/run_celeba_resume_baselines.sh`
  - it now searches only `save/models/...`
- Stopped new prediction outputs from being redirected into `outputs/save/pred`:
  - removed the `save/pred -> ../outputs/save/pred` symlink
  - created a real `save/pred/` directory
  - copied the existing prediction files from `outputs/save/pred/` into `save/pred/`
- Normalized model discovery under `save/models/`:
  - copied the old ImageNet model files from `outputs/save/models/` into `save/models/`
  - avoided leaving model symlinks in place, so future writes do not bounce back into `outputs/save/models`

### Validation

- [OK] Literal `outputs/save` references in executable code:
  - `rg -n "outputs/save" . --glob '!**/*.md' --glob '!**/*.html' --glob '!**/*.log' --glob '!**/*.json' --glob '!**/*.csv' --glob '!**/*.npy' --glob '!**/*.txt'`
  - returned no matches after the cleanup
- [OK] No symlinks remain under `save/`:
  - `find save -type l -ls`
- [OK] Prediction directory migrated:
  - `save/pred` now contains `678` files
- [OK] Model directory normalized:
  - `save/models` now contains `6` real files
- [OK] `git diff --check`

### Important Notes

- The repo still keeps `utils/path_utils.py` with a legacy read fallback for `outputs/save` so older moved artifacts can still be found if they were not copied yet.
- The important behavior change is that active write paths now resolve to `save/`, not `outputs/save/`.

### Status

[OK] **Completed**

---

## Session 41: Add completed ImageNet10 BADGE results to `report.md`

**Date**: 2026-04-16
**Task**: report-badge-results

### Summary

The user noticed that the four ImageNet10 BADGE resume runs appeared to be finished and asked for the report to be updated. Verified that all four BADGE jobs had exited, extracted the final held-out metrics from the logs, and added a new `04/16/2026` section to `report.md` summarizing the ratio sweep over `sub_ratio = 0.1, 0.3, 0.5, 0.7`.

### Main Findings

- All four BADGE runs produced usable held-out test metrics before failing in post-processing.
- Extracted main results:
  - `sub_ratio=0.1`
    - best val acc: `0.7208`
    - test acc: `0.7323`
    - AL selection time: `272.65 s`
  - `sub_ratio=0.3`
    - best val acc: `0.7215`
    - test acc: `0.7423`
    - AL selection time: `787.57 s`
    - end-to-end time: `1308.12 s`
  - `sub_ratio=0.5`
    - best val acc: `0.7223`
    - test acc: `0.7469`
    - AL selection time: `1281.51 s`
    - end-to-end time: `1654.63 s`
  - `sub_ratio=0.7`
    - best val acc: `0.7215`
    - test acc: `0.7477`
    - AL selection time: `1603.43 s`
    - end-to-end time: `1835.02 s`
- Test accuracy increased monotonically with the retained original-data ratio:
  - `0.7323 -> 0.7423 -> 0.7469 -> 0.7477`
- At the directly comparable `sub_ratio=0.3` resume setting, BADGE outperformed the earlier 2/27 resume-table entries recorded in `report.md`, including the previously cited `Hybrid` value `0.7400`.

### Failure Boundary

- The metrics are still valid because each run failed only after the evaluation lines were printed:
  - ratio `0.1`: crashed while saving predictions because `save/pred/` was missing
  - ratios `0.3`, `0.5`, `0.7`: crashed during diagnostics because `save/attribute10.json` was missing

### Files Updated

- `report.md`

### Validation

- [OK] confirmed the launch PIDs are no longer alive
- [OK] extracted the final metric lines from:
  - `logs_imagenet/resume_lr5_epoch15_badge_resnet_imagenet_1000_ratio0.1.log`
  - `logs_imagenet/resume_lr5_epoch15_badge_resnet_imagenet_1000_ratio0.3.log`
  - `logs_imagenet/resume_lr5_epoch15_badge_resnet_imagenet_1000_ratio0.5.log`
  - `logs_imagenet/resume_lr5_epoch15_badge_resnet_imagenet_1000_ratio0.7.log`
- [OK] `git diff --check`

### Status

[OK] **Completed**

---

## Session 40: Merge duplicate Active Learning Markdown docs

**Date**: 2026-04-16
**Task**: merge-active-learning-docs

### Summary

The user asked whether `md/README_ActiveLearning.md` and `md/README_ActiveLearning_Strategies.md` should be merged. After comparing both files, the strategy guide was clearly the richer and more current document, so it was kept as the canonical Active Learning doc. Merged the small amount of still-useful legacy overview content from the old file into the strategy guide, then removed the duplicate overview file.

### Main Changes

- Kept:
  - `md/README_ActiveLearning_Strategies.md`
- Removed:
  - `md/README_ActiveLearning.md`
- Updated the surviving document to:
  - rename the title from "Active Learning Strategy Usage Guide" to "Active Learning Guide"
  - state explicitly that it is now the single canonical Active Learning document
  - retain the current `alpred.py` strategy documentation
  - absorb the useful legacy content about older round-based demos and the CelebA-oriented layout

### Validation

- [OK] `git diff --check`
- [OK] `rg -n "README_ActiveLearning\\.md|README_ActiveLearning_Strategies\\.md" -S .`
  - no live references to the removed Markdown file remain

### Important Notes

- The generated HTML file `md/README_ActiveLearning_Strategies.html` still points to the surviving Markdown source and does not require changes for this merge.
- `report.md` was not updated because this change only restructures documentation.

### Status

[OK] **Completed**

---

## Session 39: Launch ImageNet10 BADGE resume sweep and harden `alpred.py` runtime paths

**Date**: 2026-04-16
**Task**: imagenet-badge-launch-and-runtime-fixes

### Summary

The user asked to run BADGE on the ImageNet10 setup using the same resume-from-base recipe as the earlier Active Learning experiments, with one `sub_ratio` per GPU: `0.1`, `0.3`, `0.5`, and `0.7` on GPUs `0`, `1`, `2`, and `3`. Before the runs could start, `alpred.py` failed in two unrelated runtime paths: a top-level import of `utils.llm_selection` crashed even when `--use_llm_selection False`, and ImageNet split indices were looked up only under `save/` even though the current repo state stores them in `outputs/save/`. Fixed both issues, then launched four detached BADGE runs and verified that all four Python processes remained alive and entered BADGE feature extraction.

### Main Changes

- Hardened `alpred.py` against optional LLM dependency failures:
  - replaced the top-level `from utils.llm_selection import llm_data_selection`
  - added lazy import helper `_require_llm_data_selection()`
  - now only raises an import error when `--use_llm_selection True`
- Hardened ImageNet split resolution in `alpred.py`:
  - added `_resolve_existing_path()`
  - ImageNet now searches split files in:
    - `save/`
    - `outputs/save/`
  - verbose runs print which directory is used when the fallback path is taken

### Commands and Artifacts

- Front-run probe that confirmed the run now reaches BADGE selection:
  - `timeout 20s env MPLCONFIGDIR=/tmp/matplotlib-safefix CUDA_VISIBLE_DEVICES=0 /people/cs/o/oxx220000/.conda/envs/ad/bin/python -u alpred.py --dataset imagenet --model resnet --num_added_img 1000 --attributes pinkcolor_fabrictexture --num_epochs 15 --verbose True --resume_model weights/model_imagenet_resnet_base.pth --use_active_learning True --sub_ratio 0.1 --al_strategy badge --al_embedding_dim 256`
- Detached BADGE runs launched with `setsid + nohup`:
  - ratio `0.1`, GPU `0`
    - PID: `2634097`
    - log: `logs_imagenet/resume_lr5_epoch15_badge_resnet_imagenet_1000_ratio0.1.log`
    - pid file: `logs_imagenet/pids/resume_lr5_epoch15_badge_ratio0.1.pid`
  - ratio `0.3`, GPU `1`
    - PID: `2634921`
    - log: `logs_imagenet/resume_lr5_epoch15_badge_resnet_imagenet_1000_ratio0.3.log`
    - pid file: `logs_imagenet/pids/resume_lr5_epoch15_badge_ratio0.3.pid`
  - ratio `0.5`, GPU `2`
    - PID: `2634930`
    - log: `logs_imagenet/resume_lr5_epoch15_badge_resnet_imagenet_1000_ratio0.5.log`
    - pid file: `logs_imagenet/pids/resume_lr5_epoch15_badge_ratio0.5.pid`
  - ratio `0.7`, GPU `3`
    - PID: `2634936`
    - log: `logs_imagenet/resume_lr5_epoch15_badge_resnet_imagenet_1000_ratio0.7.log`
    - pid file: `logs_imagenet/pids/resume_lr5_epoch15_badge_ratio0.7.pid`

### Main Findings

- The previous runtime blocker was not BADGE itself:
  - `alpred.py` crashed at import time because `utils.llm_selection` required a `transformers` symbol (`Qwen2_5_VLForConditionalGeneration`) unavailable in the current `ad` environment
- The current ImageNet split files are present under:
  - `outputs/save/train_idx_imagenet.npy`
  - `outputs/save/val_idx_imagenet.npy`
  - `outputs/save/test_idx_imagenet.npy`
- All four launched BADGE runs entered the expected setup path:
  - ImageNet split fallback resolved correctly
  - generated image directory resolved to `data/imagenet_pinkcolor_fabrictexture`
  - BADGE selection started with the expected budgets:
    - `0.1 -> 1040 / 10400`
    - `0.3 -> 3120 / 10400`
    - `0.5 -> 5200 / 10400`
    - `0.7 -> 7279 / 10400`
- GPU occupancy after launch:
  - GPUs `0`, `1`, `3` each showed the new BADGE Python process using about `1588 MiB`
  - GPU `2` already had another user process:
    - PID `2629402`
    - memory about `9054 MiB`
  - the BADGE ratio-`0.5` run still launched successfully on GPU `2` and allocated about `1588 MiB`

### Validation

- [OK] Syntax check:
  - `python3 -m py_compile alpred.py`
- [OK] Front-run smoke test reached BADGE embedding extraction without crashing
- [OK] Detached processes survived shell exit:
  - `ps -fp 2634097 2634921 2634930 2634936`
- [OK] GPU visibility confirmed:
  - `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`

### Important Notes

- The detached launch required `setsid + nohup`; plain backgrounding from the command runner did not keep the Python workers alive.
- `report.md` was not updated in this session because no BADGE result metrics are complete yet.

### Next Steps

- Monitor the four BADGE logs until training and evaluation finish.
- Extract the final val / test metrics from each ratio-specific log.
- Update `report.md` with the completed ImageNet10 BADGE results and compare them against the existing resume-from-base baselines.

### Status

[OK] **In Progress**

---

## Session 38: Implement BADGE in `alpred.py` and refresh Active Learning docs

**Date**: 2026-04-16
**Task**: alpred-badge-and-doc-refresh

### Summary

The user asked for another popular Active Learning method beyond core-set, approved `badge`, and then requested that the implementation and the current understanding of `alpred.py` be documented in the repo Markdown files. Added a BADGE selector to the main `alpred.py` Active Learning path, using the repo's existing pretrained embedding extraction plus a lightweight validation-centroid proxy classifier to form BADGE-style gradient embeddings before batch selection. Updated the main `README.md`, the Active Learning strategy guide, the legacy Active Learning overview, and `report.md` so the docs now explain how `alpred.py` dispatches selection, where `badge` fits, and why the `attribute` strategy is methodologically different from standard AL baselines.

### Main Changes

- Added BADGE support to the main CLI in:
  - `alpred.py`
- Added reusable target extraction helpers and a new BADGE implementation in:
  - `utils/active_learning.py`
- Updated high-level and AL-specific docs in:
  - `README.md`
  - `report.md`
  - `md/README_ActiveLearning.md`
  - `md/README_ActiveLearning_Strategies.md`

### Main Findings

- `alpred.py` now exposes `badge` directly through `--al_strategy`.
- The BADGE implementation is adapted to this repo's pre-training-first selection flow:
  - train and validation embeddings are extracted using the same pretrained feature models already used by `coreset` / `diversity`
  - validation embeddings plus validation labels define per-class centroids
  - centroid-induced class probabilities drive BADGE-style gradient embeddings
  - k-means++ seeding over those gradient embeddings selects the final batch
- This is intentionally different from a fully task-trained BADGE loop:
  - the repo performs subset selection before downstream fine-tuning
  - therefore a lightweight proxy classifier is needed at selection time
- The docs now explicitly distinguish three families in `alpred.py`:
  - geometry / coverage-driven
  - uncertainty-aware batch selection
  - metadata / slice-driven
- The docs now state the main Safefix-specific points of the `attribute` method:
  - class-conditional rarity
  - generated-data frequency penalty
  - log-aware failure weighting
  - deficit-based class balancing

### Validation

- [OK] Syntax check:
  - `python3 -m py_compile alpred.py utils/active_learning.py`
- [OK] In-memory BADGE sanity probe in the `ad` environment:
  - `/people/cs/o/oxx220000/.conda/envs/ad/bin/python -c "import numpy as np; from utils.active_learning import BadgeStrategy; ..."`
  - returned a valid 5-index selection and populated `last_proxy_stats`
- [OK] `git diff --check`

### Important Notes

- The default shell `python3` environment currently has a broken local PyTorch/CUDA stack:
  - importing `torch` failed with `libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1`
- Because of that environment issue, runtime verification that imports `torch` should use the working conda environment instead:
  - `/people/cs/o/oxx220000/.conda/envs/ad/bin/python`
- The BADGE implementation in this repo is best described as:
  - BADGE-style batch selection over pretrained embeddings with a validation-centroid proxy classifier
- No benchmark run was launched in this session, so the docs position BADGE as a new baseline / comparison method rather than claiming measured wins.

### Next Steps

- If the user wants empirical comparison next, run `badge` against:
  - `attribute`
  - `hybrid`
  - `stratified`
- If stronger BADGE fidelity is needed later, replace the centroid proxy with a lightweight trained probe head before selection.

### Status

[OK] **Completed**

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
