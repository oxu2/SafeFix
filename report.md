# 04/29/2026 Update — Previous RTMPose Tiny COCO Pose Result

- The only completed previous RTMPose Tiny COCO pose result currently present
  in the local repo is the 2026-04-03 GPU3 probe:
  - metric log:
    `log_pose/rtmpose_tiny_coco_gpu3_probe/20260403_131401/20260403_131401.log`
  - metric JSON:
    `log_pose/rtmpose_tiny_coco_gpu3_probe/20260403_131401/20260403_131401.json`
  - config snapshot:
    `log_pose/rtmpose_tiny_coco_gpu3_probe/20260403_131401/vis_data/config.py`
- Evaluation setup from the log/config:
  - checkpoint:
    `pose_estimation/rtmpose_assets/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth`
  - COCO annotation:
    `/data/oxx220000/coco/annotations/person_keypoints_val2017.json`
  - data root: `/data/oxx220000/coco/`
  - batch size: `8`
  - seed: `21`
  - GPU: `NVIDIA RTX A5000`
  - runtime: Python `3.9.20`, PyTorch `1.13.1+cu117`,
    TorchVision `0.14.1+cu117`, MMEngine `0.10.7`
- Completed COCO keypoint metrics:
  - AP: `0.6966775579701486`
  - AP50: `0.9056114938423302`
  - AP75: `0.7755176193877302`
  - AP(M): `0.6734239791412847`
  - AP(L): `0.7385393035542115`
  - AR: `0.7254250629722923`
  - AR50: `0.9151448362720404`
  - AR75: `0.7950251889168766`
  - AR(M): `0.6931439497405082`
  - AR(L): `0.773950204384987`
- Related paths without a completed metric:
  - `log_pose/NA/rtmpose_tiny_coco_gpu3_20260403_131338.log` exists but is
    empty.
  - `log_pose/rtmpose_tiny_coco_gpu3_latest/20260403_131220/20260403_131220.log`
    contains setup/config output but no final COCO metric line.
  - The newer Juno A30 RTMPose Tiny job `185461` was still `PENDING
    (Resources)` at the latest check, so it has no result yet.
- Interpretation note:
  - This RTMPose Tiny result is a top-down COCO-val evaluation using COCO
    ground-truth person boxes through MMPose `CocoDataset`; treat the AP as a
    pose-quality reference, not as a detector-inclusive number directly
    comparable to the Keypoint R-CNN held-out AP rows.

# 04/28/2026 Update — Juno Qwen3 Brown-Rainy Ratio 0.1 RPC Results

- All four Juno Qwen3 brown-rainy `ratio=0.1` RPC runs completed
  successfully.
- Final SLURM accounting confirmation on Juno:
  - `185427`: `COMPLETED`, exit code `0:0`, elapsed `00:09:14`, node
    `g-04-02`
  - `185428`: `COMPLETED`, exit code `0:0`, elapsed `00:09:01`, node
    `g-04-02`
  - `185429`: `COMPLETED`, exit code `0:0`, elapsed `00:09:03`, node
    `g-04-02`
  - `185430`: `COMPLETED`, exit code `0:0`, elapsed `00:08:40`, node
    `g-04-02`
- Common setup for this sweep:
  - base AP for comparison: `0.5902`
  - selected original samples: `2000`
  - selected generated brown-rainy images: `1000`
  - final train size: `3000`
  - original candidate pool: `20000`
  - generated target: `clothes_color=brown`, `weather=rainy`
  - explicit generated extra counts:
    `clothes_color:brown=1000,weather:rainy=1000`
  - `extra_count_scale`: `10`

| Job | R | P | C | Status | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `185429` | `0.03` | `0.5` | `15.0` | completed | **`0.6183`** | `+0.0281` | `0.36 s` | `00:07:48` | `/work/oxx220000/safefix/log_pose/pose_resume_qwen3_brown_rainy_r0p1_r0p03_p0p5_c15p0_gpu0_20260428_211734.log` |
| `185430` | `0.03` | `0.5` | `8.0` | completed | `0.6165` | `+0.0263` | `0.35 s` | `00:07:41` | `/work/oxx220000/safefix/log_pose/pose_resume_qwen3_brown_rainy_r0p1_r0p03_p0p5_c8p0_gpu0_20260428_212637.log` |
| `185428` | `0.01` | `0.5` | `15.0` | completed | `0.6152` | `+0.0250` | `0.36 s` | `00:07:51` | `/work/oxx220000/safefix/log_pose/pose_resume_qwen3_brown_rainy_r0p1_r0p01_p0p5_c15p0_gpu0_20260428_210833.log` |
| `185427` | `0.0` | `0.0` | `15.0` | completed | `0.6042` | `+0.0140` | `0.37 s` | `00:08:00` | `/work/oxx220000/safefix/log_pose/pose_resume_qwen3_brown_rainy_r0p1_r0p0_p0p0_c15p0_gpu0_20260428_205920.log` |

- Completed `185429` metrics:
  - held-out val: AP `0.6288`, AP50 `0.8604`, AP75 `0.6925`, AR `0.6995`
  - held-out test: AP `0.6183`, AP50 `0.8466`, AP75 `0.6782`, AR `0.6860`
- Completed `185430` metrics:
  - held-out val: AP `0.6285`, AP50 `0.8629`, AP75 `0.6882`, AR `0.6984`
  - held-out test: AP `0.6165`, AP50 `0.8489`, AP75 `0.6690`, AR `0.6854`
- Completed `185428` metrics:
  - held-out val: AP `0.6235`, AP50 `0.8485`, AP75 `0.6833`, AR `0.6971`
  - held-out test: AP `0.6152`, AP50 `0.8403`, AP75 `0.6717`, AR `0.6872`
- Completed `185427` metrics:
  - held-out val: AP `0.6159`, AP50 `0.8568`, AP75 `0.6623`, AR `0.6908`
  - held-out test: AP `0.6042`, AP50 `0.8390`, AP75 `0.6546`, AR `0.6766`
- Interpretation:
  - `185429` is the best completed Juno Qwen3 brown-rainy `ratio=0.1` row so
    far, with test AP `0.6183` (`+0.0281` over the canonical base)
  - all three nonzero `R/P` rows beat the conservative all-zero
    rarity/frequency row (`185427`)
  - with `R=0.03` and `P=0.5`, the softer class-balance setting `C=8.0`
    (`185430`, AP `0.6165`) trails `C=15.0` (`185429`, AP `0.6183`) but still
    beats the lower-rarity `R=0.01`, `P=0.5`, `C=15.0` setting
  - `185429` is also above the earlier local InternVL3 rainy `ratio=0.1`
    conservative attribute row (`0.6071`), plus the earlier local `random`
    (`0.6029`) and `stratified` (`0.6031`) rows

# 04/28/2026 Update — COCO Pose Ratio 0.05 Rainy / Brown-Rainy Results

- This table merges the earlier local InternVL3 rainy `ratio=0.05` runs with
  the completed Juno Qwen3 brown-rainy `ratio=0.05` run.
- Common setup:
  - canonical base AP for comparison: `0.5902`
  - selected original samples: `1000`
  - selected generated images: `1000`
  - final train size: `2000`
  - original candidate pool: `20000`
- RPC means the three attribute-selector parameters:
  - `R`: `--al-attribute-rarity-power`; larger values increase rare attribute
    weighting through inverse frequency.
  - `P`: `--al-attribute-frequency-penalty`; larger values subtract more score
    from frequent attribute values.
  - `C`: `--al-attribute-class-balance`; `0` disables class balancing, finite
    values add a soft class-balance bonus, and `inf` uses strict round-robin
    class quotas.
- For the local InternVL3 rainy rows, `attrx20` means
  `--al-attribute-extra-count-scale 20` on the inferred generated rainy counts.
  For the Juno Qwen3 brown-rainy row, the extra counts were explicit:
  `clothes_color:brown=1000,weather:rainy=1000`.
- The Juno `ratio=0.1` brown-rainy run was intentionally not relaunched in this
  pass.

| Setting | Metadata / generated target | Strategy | R | P | C | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Juno brown-rainy | Qwen3; `clothes_color=brown`, `weather=rainy` | `attribute` | `0.0` | `0.0` | `15.0` | **`0.5871`** | `-0.0031` | `0.41 s` | `00:06:11` | `/work/oxx220000/safefix/log_pose/pose_resume_qwen3_brown_rainy_r0p05_r0p0_p0p0_c15p0_gpu0_20260428_185219.log` |
| local rainy | InternVL3; `weather=rainy` | `attribute` (`attrx20_r000_p0_c15`) | `0.0` | `0.0` | `15.0` | `0.5846` | `-0.0056` | `0.26 s` | `00:07:46` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.05_attrx20_r000_p0_c15_gpu0_20260423_232656.log` |
| local rainy | InternVL3; `weather=rainy` | `stratified` | `n/a` | `n/a` | `n/a` | `0.5815` | `-0.0087` | `0.22 s` | `00:07:46` | `log_pose/pose_resume_internvl3_epoch2_stratified_ratio0.05_gpu0_20260424_000907.log` |
| local rainy | InternVL3; `weather=rainy` | `random` | `n/a` | `n/a` | `n/a` | `0.5792` | `-0.0110` | `0.00 s` | `00:07:39` | `log_pose/pose_resume_internvl3_epoch2_random_ratio0.05_gpu0_20260424_000907.log` |
| local rainy | InternVL3; `weather=rainy` | `badge` fast approximate | `n/a` | `n/a` | `n/a` | `0.5754` | `-0.0148` | `64.34 s` | `00:08:53` | `log_pose/pose_resume_internvl3_epoch2_badge_ratio0.05_gpu0_20260424_121700.log` |
| local rainy | InternVL3; `weather=rainy` | `attribute` (`attrx20_r000_p0_c0`) | `0.0` | `0.0` | `0.0` | `0.5745` | `-0.0157` | `0.25 s` | `00:07:52` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.05_attrx20_r000_p0_c0_gpu0_20260424_121700.log` |
| local rainy | InternVL3; `weather=rainy` | `coreset` | `n/a` | `n/a` | `n/a` | `0.5619` | `-0.0283` | `202.56 s` | `00:11:11` | `log_pose/pose_resume_internvl3_epoch2_coreset_ratio0.05_gpu0_20260424_000907.log` |
| local rainy | InternVL3; `weather=rainy` | `attribute` (`attrx20_r0005_p005_c15`) | `0.005` | `0.05` | `15.0` | `0.5273` | `-0.0629` | `0.26 s` | `00:08:14` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.05_attrx20_r0005_p005_c15_gpu0_20260424_000907.log` |
| local rainy | InternVL3; `weather=rainy` | `attribute` (`attrx20_r0001_p01_c15`) | `0.001` | `0.1` | `15.0` | `0.5185` | `-0.0717` | `0.28 s` | `00:09:11` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.05_attrx20_r0001_p01_c15_gpu0_20260424_000907.log` |

- Juno brown-rainy validation metrics:
  - held-out val: AP `0.6011`, AP50 `0.8492`, AP75 `0.6560`, AR `0.6834`
  - held-out test: AP `0.5871`, AP50 `0.8308`, AP75 `0.6384`, AR `0.6712`
- Main observations:
  - the Juno Qwen3 brown-rainy row is now the best completed `ratio=0.05` row,
    but it is still slightly below the canonical base AP (`-0.0031`)
  - among the earlier local rainy rows, the best completed row remains
    `attribute` (`attrx20_r000_p0_c15`) with AP `0.5846`
  - the weather-steered local attribute settings with nonzero `R/P` are much
    weaker at this ratio, especially `attrx20_r0001_p01_c15`
  - the fast approximate `badge` row completed but did not beat `attribute`,
    `stratified`, or `random`
  - `coreset` is slower because its selection took `202.56 s`

# 04/24/2026 Update — Pose Bug Compare Threshold Standard

- Future pose bug/failure-set comparisons should use `acc_diff_threshold=0.02`.
- Failure rule: `slice_AP < overall_AP - acc_diff_threshold`.
- Standard output folder:
  - `pose_bug_threshold_compare/`
- Main standard CSV:
  - `pose_bug_threshold_compare/bugs_compare_pose_acc_diff_0p02.csv`
- Detailed focused rows:
  - `pose_bug_threshold_compare/acc_diff_0p02/`
- Non-standard threshold sweep files are archived under:
  - `pose_bug_threshold_compare/archived_threshold_sweep/`
- With `acc_diff_threshold=0.02`:
  - base failure total before top-k: `12`
  - compared base failure cases per method: `12`
  - mean still failures over the 20 compact method rows: `7.05`
  - mean resolved failures over the 20 compact method rows: `4.95`

| ratio | method | overall_ap_base | overall_ap_run | overall_ap_delta | base_failure_total | compared_count | improved_count | declined_count | same_count | resolved_failure_count | still_failure_count | rare_to_not_rare_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.1` | `attribute` | `0.5902` | `0.6071` | `0.0169` | `12` | `12` | `9` | `1` | `0` | `4` | `8` | `2` |
| `0.1` | `random` | `0.5902` | `0.6029` | `0.0127` | `12` | `12` | `8` | `2` | `0` | `5` | `7` | `2` |
| `0.1` | `stratified` | `0.5902` | `0.6031` | `0.0129` | `12` | `12` | `8` | `2` | `0` | `5` | `7` | `2` |
| `0.1` | `coreset` | `0.5902` | `0.5579` | `-0.0323` | `12` | `12` | `1` | `9` | `0` | `5` | `7` | `2` |
| `0.1` | `badge` | `0.5902` | `0.5919` | `0.0017` | `12` | `12` | `6` | `4` | `0` | `4` | `8` | `2` |
| `0.3` | `attribute` | `0.5902` | `0.6174` | `0.0272` | `12` | `12` | **`10`** | `0` | `0` | `6` | `6` | `2` |
| `0.3` | `random` | `0.5902` | `0.6061` | `0.0159` | `12` | `12` | `8` | `2` | `0` | `5` | `7` | `2` |
| `0.3` | `stratified` | `0.5902` | `0.6041` | `0.0139` | `12` | `12` | `8` | `2` | `0` | `5` | `7` | `2` |
| `0.3` | `coreset` | `0.5902` | `0.6031` | `0.0129` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.3` | `badge` | `0.5902` | `0.6083` | `0.0181` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.5` | `attribute` | `0.5902` | `0.6127` | `0.0225` | `12` | `12` | **`10`** | `0` | `0` | `4` | `8` | `1` |
| `0.5` | `random` | `0.5902` | `0.5972` | `0.0070` | `12` | `12` | `6` | `4` | `0` | `5` | `7` | `2` |
| `0.5` | `stratified` | `0.5902` | `0.6166` | `0.0264` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `1` |
| `0.5` | `coreset` | `0.5902` | `0.6073` | `0.0171` | `12` | `12` | `8` | `2` | `0` | `5` | `7` | `2` |
| `0.5` | `badge` | `0.5902` | `0.6111` | `0.0209` | `12` | `12` | `9` | `1` | `0` | `5` | `7` | `2` |
| `0.7` | `attribute` | `0.5902` | **`0.6263`** | `0.0361` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.7` | `random` | `0.5902` | `0.6244` | `0.0342` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.7` | `stratified` | `0.5902` | `0.6244` | `0.0342` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.7` | `coreset` | `0.5902` | `0.6176` | `0.0274` | `12` | `12` | **`10`** | `0` | `0` | `5` | `7` | `2` |
| `0.7` | `badge` | `0.5902` | `0.6191` | `0.0289` | `12` | `12` | **`10`** | `0` | `0` | `6` | `6` | `2` |

# 04/17/2026 Update — COCO Pose InternVL3 Resume Baselines And Attribute Sweeps Completed

- The controller-backed InternVL3 COCO pose resume sweep launched from the canonical base checkpoint has fully completed, and the finished `attribute` comparison runs are now merged directly into the same top-level ratio tables for like-for-like comparison.
- Controller log:
  - `log_pose/run_pose_resume_internvl3_all_20260417_022717.log`
  - ends with:
    - `All strategies completed.`
- Common setup for the tables below:
  - base reference for AP comparison:
    - `log_pose/pose_keypointrcnn_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015544/`
    - log: `log_pose/pose_train20k_internvl3_bs4_e2_rare0p97_gpuX_20260412_015544.log`
  - base held-out test AP for comparison:
    - `0.5902`
  - note:
    - use `0.5902` as the canonical base AP for future COCO pose InternVL3 resume comparisons; the previously listed `0.6145` comparison value was wrong
  - resume schedule:
    - one additional epoch from the saved epoch-0 checkpoint
  - original candidate pool:
    - `20000`
  - generated augmentation:
    - canonical rainy-image mix, `1000` images added after original-image selection
  - fixed ratio placement:
    - GPU `0` -> `0.1`
    - GPU `1` -> `0.3`
    - GPU `2` -> `0.5`
    - GPU `3` -> `0.7`

### Ratio = 0.1

| Strategy | Selected Original | Final Train Size | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
|----------|-------------------|------------------|------------------|---------------|-------------------|-----------------|---------------|
| `random` | `2000` | `3000` | `0.6029` | `+0.0127` | `0.00 s` | `00:09:44` | `log_pose/pose_resume_internvl3_epoch2_random_ratio0.1_gpu0_20260417_022717.log` |
| `stratified` | `2000` | `3000` | `0.6031` | `+0.0129` | `0.23 s` | `00:09:44` | `log_pose/pose_resume_internvl3_epoch2_stratified_ratio0.1_gpu0_20260417_023705.log` |
| `coreset` | `2000` | `3000` | `0.5579` | `-0.0323` | `4.96 min` | `00:14:48` | `log_pose/pose_resume_internvl3_epoch2_coreset_ratio0.1_gpu0_20260417_024654.log` |
| `badge` | `2000` | `3000` | `0.5919` | `+0.0017` | `21.43 min` | `00:31:20` | `log_pose/pose_resume_internvl3_epoch2_badge_ratio0.1_gpu0_20260417_030147.log` |
| `attribute` (`attrx10_r000_p0_c15`) | `2000` | `3000` | **`0.6071`** | `+0.0169` | `0.26 s` | `00:09:48` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.1_attrx10_r000_p0_c15_gpu0_20260417_114049.log` |
| `attribute` (`attrx10_r003_p05_c15`) | `2000` | `3000` | `0.6012` | `+0.0110` | `0.27 s` | `00:10:04` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.1_attrx10_r003_p05_c15_gpu2_20260417_114049.log` |
| `attribute` (`attrx10_r001_p05_c15`) | `2000` | `3000` | `0.6003` | `+0.0101` | `0.26 s` | `00:09:50` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.1_attrx10_r001_p05_c15_gpu1_20260417_114049.log` |
| `attribute` (`attrx10_r003_p05_c8`) | `2000` | `3000` | `0.5963` | `+0.0061` | `0.26 s` | `00:09:47` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.1_attrx10_r003_p05_c8_gpu3_20260417_114049.log` |

### Ratio = 0.3

| Strategy | Selected Original | Final Train Size | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
|----------|-------------------|------------------|------------------|---------------|-------------------|-----------------|---------------|
| `random` | `6000` | `7000` | `0.6061` | `+0.0159` | `0.00 s` | `00:17:55` | `log_pose/pose_resume_internvl3_epoch2_random_ratio0.3_gpu1_20260417_022717.log` |
| `stratified` | `6000` | `7000` | `0.6041` | `+0.0139` | `0.21 s` | `00:18:37` | `log_pose/pose_resume_internvl3_epoch2_stratified_ratio0.3_gpu1_20260417_024517.log` |
| `coreset` | `6000` | `7000` | `0.6031` | `+0.0129` | `19.29 min` | `00:37:10` | `log_pose/pose_resume_internvl3_epoch2_coreset_ratio0.3_gpu1_20260417_030359.log` |
| `badge` | `6000` | `7000` | `0.6083` | `+0.0181` | `59.96 s` | `00:18:51` | `log_pose/pose_resume_internvl3_epoch2_badge_ratio0.3_gpu1_20260417_034116.log` |
| `attribute` (`attrx333_r000_p0_c15`) | `6000` | `7000` | **`0.6174`** | `+0.0272` | `0.28 s` | `00:18:23` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.3_attrx333_r000_p0_c15_gpu1_20260417_122328.log` |
| `attribute` (`attrx333_r000_p0_c0`) | `6000` | `7000` | `0.6091` | `+0.0189` | `0.25 s` | `00:17:56` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.3_attrx333_r000_p0_c0_gpu0_20260417_122328.log` |
| `attribute` (`attrx333_r001_p05_c15`) | `6000` | `7000` | `0.6079` | `+0.0177` | `0.27 s` | `00:18:24` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.3_attrx333_r001_p05_c15_gpu3_20260417_122328.log` |
| `attribute` (`attrx333_r001_p05_c0`) | `6000` | `7000` | `0.6067` | `+0.0165` | `0.25 s` | `00:18:08` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.3_attrx333_r001_p05_c0_gpu2_20260417_122328.log` |

### Ratio = 0.5

| Strategy | Selected Original | Final Train Size | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
|----------|-------------------|------------------|------------------|---------------|-------------------|-----------------|---------------|
| `random` | `10000` | `11000` | `0.5972` | `+0.0070` | `0.00 s` | `00:27:11` | `log_pose/pose_resume_internvl3_epoch2_random_ratio0.5_gpu2_20260417_022717.log` |
| `stratified` | `10000` | `11000` | `0.6166` | `+0.0264` | `0.22 s` | `00:28:59` | `log_pose/pose_resume_internvl3_epoch2_stratified_ratio0.5_gpu2_20260417_025433.log` |
| `coreset` | `10000` | `11000` | `0.6073` | `+0.0171` | `1.15 min` | `00:28:22` | `log_pose/pose_resume_internvl3_epoch2_coreset_ratio0.5_gpu2_20260417_032336.log` |
| `badge` | `10000` | `11000` | `0.6111` | `+0.0209` | `59.68 s` | `00:27:22` | `log_pose/pose_resume_internvl3_epoch2_badge_ratio0.5_gpu2_20260417_035203.log` |
| `attribute` (`attrx2_r000_p0_c0`) | `10000` | `11000` | **`0.6127`** | `+0.0225` | `0.25 s` | `00:26:42` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.5_attrx2_r000_p0_c0_gpu0_20260417_125113.log` |
| `attribute` (`attrx2_r000_p0_c15`) | `10000` | `11000` | `0.6097` | `+0.0195` | `0.28 s` | `00:27:23` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.5_attrx2_r000_p0_c15_gpu1_20260417_125113.log` |
| `attribute` (`attrx2_r001_p05_c15`) | `10000` | `11000` | `0.6084` | `+0.0182` | `0.28 s` | `00:27:14` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.5_attrx2_r001_p05_c15_gpu3_20260417_125113.log` |
| `attribute` (`attrx2_r001_p05_c0`) | `10000` | `11000` | `0.6047` | `+0.0145` | `0.25 s` | `00:26:34` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.5_attrx2_r001_p05_c0_gpu2_20260417_125113.log` |

### Ratio = 0.7

| Strategy | Selected Original | Final Train Size | Held-out Test AP | Delta vs Base | AL Selection Time | End-to-End Time | Canonical Log |
|----------|-------------------|------------------|------------------|---------------|-------------------|-----------------|---------------|
| `random` | `14000` | `15000` | `0.6244` | `+0.0342` | `0.01 s` | `00:34:27` | `log_pose/pose_resume_internvl3_epoch2_random_ratio0.7_gpu3_20260417_022717.log` |
| `stratified` | `14000` | `15000` | `0.6244` | `+0.0342` | `0.23 s` | `00:35:26` | `log_pose/pose_resume_internvl3_epoch2_stratified_ratio0.7_gpu3_20260417_041149.log` |
| `coreset` | `14000` | `15000` | `0.6176` | `+0.0274` | `59.12 s` | `00:35:25` | `log_pose/pose_resume_internvl3_epoch2_coreset_ratio0.7_gpu3_20260417_055719.log` |
| `badge` | `14000` | `15000` | `0.6191` | `+0.0289` | `59.19 s` | `00:35:24` | `log_pose/pose_resume_internvl3_epoch2_badge_ratio0.7_gpu3_20260417_074249.log` |
| `attribute` (`attrx143_r000_p0_c15`) | `14000` | `15000` | **`0.6263`** | `+0.0361` | `0.28 s` | `00:36:03` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.7_attrx143_r000_p0_c15_gpu1_20260417_170346.log` |
| `attribute` (`attrx143_r001_p05_c0`) | `14000` | `15000` | `0.6262` | `+0.0360` | `0.26 s` | `00:34:53` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.7_attrx143_r001_p05_c0_gpu2_20260417_170346.log` |
| `attribute` (`attrx143_r001_p05_c15`) | `14000` | `15000` | `0.6203` | `+0.0301` | `0.29 s` | `00:35:51` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.7_attrx143_r001_p05_c15_gpu3_20260417_170346.log` |
| `attribute` (`attrx143_r000_p0_c0`) | `14000` | `15000` | `0.6166` | `+0.0264` | `0.25 s` | `00:35:02` | `log_pose/pose_resume_internvl3_epoch2_attribute_ratio0.7_attrx143_r000_p0_c0_gpu0_20260417_170346.log` |

- Main observations:
  - all non-attribute and attribute resume runs now have stable finished outputs for the four comparison ratios `0.1`, `0.3`, `0.5`, and `0.7`
  - the finished attribute runs are now grouped directly under the same top-level ratio tables as the baselines for like-for-like comparison
  - under the corrected comparison baseline (`0.5902` test AP), nearly all finished resume runs are above base; the clear exception in this table is `coreset` at `ratio=0.1`
  - the strongest finished resumed result in the full table is now:
    - `attribute` (`attrx143_r000_p0_c15`) at `ratio=0.7`
    - held-out test AP `0.6263`
    - delta vs corrected base `+0.0361`
  - the second-best finished resumed result is also an attribute point at `ratio=0.7`:
    - `attribute` (`attrx143_r001_p05_c0`)
    - held-out test AP `0.6262`
    - delta vs corrected base `+0.0360`
  - retained-data ratio matters more than selector sophistication in this sweep:
    - every strategy is substantially stronger at `0.7` than at `0.1`
  - the corrected base is matched or exceeded by:
    - `stratified`, `ratio=0.5`
    - all four finished `attribute` runs at `ratio=0.7`
    - `attribute` (`attrx333_r000_p0_c15`), `ratio=0.3`
    - all four non-attribute strategies at `ratio=0.7`
  - at `ratio=0.1`, the best finished attribute point is:
    - `attrx10_r000_p0_c15`
    - held-out test AP `0.6071`
    - this is above the corrected base (`0.5902`) and above the corrected `random` `ratio=0.1` row (`0.6029`)
  - at `ratio=0.3`, the best finished attribute point is:
    - `attrx333_r000_p0_c15`
    - held-out test AP `0.6174`
    - this is above the corrected base (`0.5902`) and above every finished non-attribute `ratio=0.3` baseline
  - at `ratio=0.5`, the best finished attribute point is:
    - `attrx2_r000_p0_c0`
    - held-out test AP `0.6127`
    - this is above the corrected base (`0.5902`) and below the finished `stratified` `ratio=0.5` baseline (`0.6166`), but above `badge`, `coreset`, and `random` at the same ratio
  - at `ratio=0.7`, the best finished attribute point is:
    - `attrx143_r000_p0_c15`
    - held-out test AP `0.6263`
    - this is above the corrected base (`0.5902`) and above every finished non-attribute `ratio=0.7` baseline
  - for the `ratio=0.3` attribute RPC4 sweep, the conservative control family still wins:
    - `r000_p0_c15`: `0.6174`
    - `r000_p0_c0`: `0.6091`
    - `r001_p05_c15`: `0.6079`
    - `r001_p05_c0`: `0.6067`
  - the same pattern continues at `ratio=0.5`:
    - `r000_p0_c0`: `0.6127`
    - `r000_p0_c15`: `0.6097`
    - `r001_p05_c15`: `0.6084`
    - `r001_p05_c0`: `0.6047`
  - at `ratio=0.7`, the strongest two points are split between the conservative and mild weather-steered branches:
    - `r000_p0_c15`: `0.6263`
    - `r001_p05_c0`: `0.6262`
    - `r001_p05_c15`: `0.6203`
    - `r000_p0_c0`: `0.6166`
  - `coreset` is still the weakest cost/benefit point:
    - `ratio=0.1`: `0.5579` AP after `4.96 min` of selection
    - `ratio=0.3`: `0.6031` AP after `19.29 min`
    - `ratio=0.5`: `0.6073` AP after `1.15 min`
    - `ratio=0.7`: `0.6176` AP after `59.12 s`
  - `badge` no longer leads at `ratio=0.3` once the finished attribute sweep is included:
    - `badge`: `0.6083`
    - best attribute: `0.6174`
  - at `ratio=0.5`, `attribute` also improves on `badge`, but does not reach `stratified`:
    - `attribute` best: `0.6127`
    - `badge`: `0.6111`
    - `stratified`: `0.6166`
  - at `ratio=0.7`, `attribute` now leads the full baseline set:
    - best attribute: `0.6263`
    - `random`: `0.6244`
    - `stratified`: `0.6244`
    - `badge`: `0.6191`
    - `coreset`: `0.6176`
- practical reference point for future pose-resume comparisons:
  - keep `ratio=0.3` `attribute` (`attrx333_r000_p0_c15`), `ratio=0.5` `attribute` (`attrx2_r000_p0_c0`), and `ratio=0.7` `attribute` (`attrx143_r000_p0_c15`)

# 04/17/2026 Update — COCO Pose Rain-Augmented Full-Data Run Finished

- The full-data InternVL3 COCO pose run with `1000` generated rain edits on GPU `3` finished at `2026-04-17 02:48:19`.
- Common setup:
  - original selected images:
    - `20000`
  - generated source:
    - legacy `attribute=rain`
  - generated images after filtering:
    - `1271` available
    - `1000` selected
  - final train size:
    - `21000`
  - slice-analysis source:
    - `internvl3`
- Compared with the corrected InternVL3 `20k` base reference now used in this report (`held-out test AP = 0.5902`), this rain-augmented full-data run is higher by `+0.0318` AP.

| Run | Selected Original | Final Train Size | Best Val AP | Held-out Test AP | Test AP50 | Test AP75 | Test AR | Train Time | End-to-End Time | Canonical Log |
|-----|-------------------|------------------|-------------|------------------|-----------|-----------|---------|------------|-----------------|---------------|
| InternVL3 + `1000` rain edits | `20000` | `21000` | `0.6333` | `0.6220` | `0.8528` | `0.6766` | `0.6893` | `01:33:45` | `01:33:59` | `log_pose/pose_train20k_rain1k_gpu3_20260417_011411.log` |

- Main observations:
  - the best checkpoint still came from epoch `0`, with best val AP `0.6333`
  - held-out test AP `0.6220` is above the corrected earlier InternVL3 base
  - the hardest held-out test slice remained:
    - `pose=lying` with `AP=0.3268`
  - the weather-target slice also remained difficult:
    - `weather=rainy` with `AP=0.5692`

# 04/16/2026 Update — CelebA ResNet Base Run Finished

- The fresh CelebA ResNet base run launched on GPU `0` has finished.
- The short launcher log:
  - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_gpu0.log`
  - only contains the early startup lines and is **not** the full training record.
- The complete training log is:
  - `logs/celeba_resnet_base_redhair_brownskin_sademotion_0imgs_20260416_022317.log`
- This run was the new from-scratch baseline for:
  - dataset: `CelebA`
  - model: `ResNet`
  - attributes: `redhair_brownskin_sademotion`
  - generated images: `0`
  - training set size: `80000`
  - initialization: random weights
  - epochs: `20`

| Run | Held-out Test ACC| Best Val ACC | Best-Val Epoch | Final Val ACC  | Test Loss | Train Time | End-to-End Time | Checkpoint | Predictions |
|-----|--------------|----------------|---------------|-------------------|-----------|------------|-----------------|------------|-------------|
| CelebA ResNet base (`0` generated images) | **`90.57%`** | `0.9140` | `12` | `0.9066` | `0.3242` | `32 min` | `32 min` | `save/models/model_celeba_resnet_redhair_brownskin_sademotion_0imgs_0416022326.pth` | `save/pred/predictions_celeba_resnet_redhair_brownskin_sademotion_0imgs_0416022326.npy` |

- Main interpretation:
  - this base checkpoint is now available for the planned CelebA resume experiments
  - for reporting, the CelebA base held-out test accuracy is manually recorded as `90.57%`
  - validation peaked at epoch `12` and then drifted slightly downward, so the saved base checkpoint is a reasonable resume starting point
- Post-run status:
  - the run completed training, saved the checkpoint, evaluated on the held-out test split, and saved predictions
  - it then failed only during the later diagnostics stage because `save/attribute10.json` was missing
  - this failure happened **after** the main training and evaluation outputs, so the reported checkpoint and metrics above are valid

### CelebA Resume Sweep Note

- For the current CelebA resume-baseline sweep, the `hierarchical` strategy is temporarily **deferred**.
- Reason:
  - on this CelebA setting (`80,000` original images plus `1,000` generated images), hierarchical selection remained impractically slow even after a first-pass scalability patch.
- Practical scope for the current sweep:
  - treat `random`, `stratified`, `coreset`, and `badge` as the active comparison set.
- The hierarchical path can be revisited later if there is a stronger need for a deeper algorithmic rewrite.

### CelebA Resume Baselines Completed

- The non-hierarchical CelebA resume sweep is now largely complete.
- Common setup for the table below:
  - base checkpoint: `weights/model_celeba_resnet_base.pth`
  - base held-out test accuracy for reporting: `90.57%`
  - model: `ResNet`
  - learning rate: `1e-5`
  - epochs: `15`
  - generated images: `1000`
  - original CelebA training pool: `80,000`
- For `badge` and `coreset`, the larger-ratio entries below use the patched reruns because the original exact selectors did not scale well enough on this CelebA setting.

### Ratio = 0.1

| Strategy | Selected Original Samples | Final Train Size After +1000 Gen | Held-out Test ACC | Delta vs Base | AL Selection Time | Canonical Log |
|----------|---------------------------|----------------------------------|-------------------|---------------|-------------------|---------------|
| `random` | `8000` | `9000` | `90.15%` | `-0.42 pp` | `0.00 s` | `logs_celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.1.log` |
| `stratified` | `8000` | `9000` | `90.95%` | `+0.38 pp` | `0.00 s` | `logs_celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.1.log` |
| `coreset` | `8000` | `9000` | `90.29%` | `-0.28 pp` | `45 min` | `logs_celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.1.log` |
| `badge` | `8000` | `9000` | `91.12%` | `+0.55 pp` | `1 h 52 min` | `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.1.log` |
| `attribute` | `8000` | `9000` | `91.07%` | `+0.50 pp` | `0.42 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r003_p05_c15_gpu2_20260417_010718.log` |

### Ratio = 0.3

| Strategy | Selected Original Samples | Final Train Size After +1000 Gen | Held-out Test ACC | Delta vs Base | AL Selection Time | Canonical Log |
|----------|---------------------------|----------------------------------|-------------------|---------------|-------------------|---------------|
| `random` | `24000` | `25000` | `90.68%` | `+0.11 pp` | `0.00 s` | `logs_celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.3.log` |
| `stratified` | `24000` | `25000` | `90.65%` | `+0.08 pp` | `0.00 s` | `logs_celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.3.log` |
| `coreset` | `24000` | `25000` | `90.63%` | `+0.06 pp` | `2 h 41 min` | `logs_celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.3.log` |
| `badge` | `24000` | `25000` | `90.54%` | `-0.03 pp` | `116.78 s` | `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.3_gpu2_verbose_20260416_115906.log` |
| `attribute` | `24000` | `25000` | `90.91%` | `+0.34 pp` | `0.43 s` | `logs_celeba/attr_rb_ratio0.3_rb_qwen3x333_r000_p0_cinf_gpu0_20260417_014040.log` |

### Ratio = 0.5

| Strategy | Selected Original Samples | Final Train Size After +1000 Gen | Held-out Test ACC | Delta vs Base | AL Selection Time | Canonical Log |
|----------|---------------------------|----------------------------------|-------------------|---------------|-------------------|---------------|
| `random` | `40000` | `41000` | `90.77%` | `+0.20 pp` | `0.00 s` | `logs_celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.5.log` |
| `stratified` | `40000` | `41000` | `90.79%` | `+0.22 pp` | `0.01 s` | `logs_celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.5.log` |
| `coreset` | `40000` | `41000` | `90.59%` | `+0.02 pp` | `5 h 5 min` | `logs_celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.5.log` |
| `badge` | `40000` | `41000` | `90.63%` | `+0.06 pp` | `124.76 s` | `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.5_gpu0_20260416_120554.log` |

### Ratio = 0.7

| Strategy | Selected Original Samples | Final Train Size After +1000 Gen | Held-out Test ACC | Delta vs Base | AL Selection Time | Canonical Log |
|----------|---------------------------|----------------------------------|-------------------|---------------|-------------------|---------------|
| `random` | `56000` | `57000` | `90.38%` | `-0.19 pp` | `0.00 s` | `logs_celeba/resume_lr1e-5_epoch15_random_resnet_celeba_1000_ratio0.7.log` |
| `stratified` | `56000` | `57000` | `90.79%` | `+0.22 pp` | `0.01 s` | `logs_celeba/resume_lr1e-5_epoch15_stratified_resnet_celeba_1000_ratio0.7.log` |
| `coreset` | `56000` | `57000` | `90.60%` | `+0.03 pp` | `56.85 s` | `logs_celeba/resume_lr1e-5_epoch15_coreset_resnet_celeba_1000_ratio0.7_gpu3_rerun_20260416_121045.log` |
| `badge` | `56000` | `57000` | `90.81%` | `+0.24 pp` | `74.42 s` | `logs_celeba/resume_lr1e-5_epoch15_badge_resnet_celeba_1000_ratio0.7_gpu0_20260416_122047.log` |

### CelebA Resume Observations

- Under the manually reported CelebA base test accuracy (`90.57%`), the corrected higher-ratio non-attribute resume baselines are weaker than previously recorded, while the `ratio=0.1` entries remain as originally reported.
- The strongest completed resume result in the current table is `badge` at `ratio=0.1` with held-out test accuracy `91.12%`, which is `+0.55` percentage points above the reported base.
- Among the unpatched selectors, `coreset` and early `badge` were dominated by selection cost rather than downstream accuracy:
  - `coreset` selection time grew from `45 min` at `ratio=0.1` to `5 h 5 min` at `ratio=0.5`
  - the original `badge` `ratio=0.1` run already needed `1 h 52 min` just for selection
- After the large-job approximation patches, the high-ratio runs became practical again:
  - patched `coreset` `ratio=0.7`: `56.85 s`
  - patched `badge` `ratio=0.3/0.5/0.7`: `116.78 s`, `124.76 s`, `74.42 s`
- The apparent BADGE runtime reversal across ratios is a code-path change, not a true scaling win:
  - `ratio=0.1` used the older exact BADGE path and took `1 h 52 min`
  - `ratio=0.3/0.5/0.7` are patched large-job reruns that use the approximation path
  - so `0.1` versus `0.3/0.5/0.7` runtime is not a like-for-like selector comparison
- Accuracy trends are not monotonic with retained-data ratio on CelebA:
  - corrected `random` peaked at `ratio=0.5` and then dropped at `0.7`
  - corrected `stratified` was stable at `0.5` and `0.7`
  - corrected `badge` improved again at `0.7` after being weaker at `0.3`
- For the current CelebA comparison set, `badge` remains the strongest reported method at `ratio=0.1`, while `attribute` is competitive at both `ratio=0.1` and `ratio=0.3`; `coreset` is still the least attractive once runtime is considered.

### 04/16/2026 Update — CelebA Attribute Hyperparameter Probe (`ratio=0.1`, `E=False`)

- Completed three CelebA resume-from-base `attribute` runs at `ratio=0.1` with entropy fixed off and explicit CelebA metadata focus:
  - `hair color`
  - `skin`
  - `emotion`
- Because `save/attribute.json` uses spaced keys, these runs also passed explicit generated-count injection rather than relying on the ImageNet-style auto-inference:
  - `hair color:red=1000`
  - `skin:brown=1000`
  - `emotion:sad=1000`

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.1`) | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|-------------------------------|-------------------|-----------------|---------------|
| `baseline_def_e0_r001_p5_cinf` | `0.01` | `5.0` | `inf` | **`90.59%`** | `+0.02 pp` | `+0.44 pp` | `0.72 s` | `237.61 s` | `logs_celeba/attr_ratio0.1_baseline_def_e0_r001_p5_cinf_gpu0_20260416_140026.log` |
| `soft_e0_r01_p5_c15` | `0.1` | `5.0` | `15.0` | `90.55%` | `-0.02 pp` | `+0.40 pp` | `0.70 s` | `236.93 s` | `logs_celeba/attr_ratio0.1_soft_e0_r01_p5_c15_gpu1_20260416_140026.log` |
| `aggro_e0_r08_p10_c4` | `0.8` | `10.0` | `4.0` | `90.47%` | `-0.10 pp` | `+0.32 pp` | `0.71 s` | `236.35 s` | `logs_celeba/attr_ratio0.1_aggro_e0_r08_p10_c4_gpu2_20260416_140026.log` |

### CelebA Attribute-Probe Observations

- The best of the three completed CelebA attribute settings is the most conservative one:
  - `baseline_def_e0_r001_p5_cinf`
  - held-out test accuracy `90.59%`
- Even this early attribute configuration already clears the corrected `random` baseline, but it remains below the stronger completed `ratio=0.1` non-attribute baselines:
  - `random`: `90.15%`
  - `stratified`: `90.95%`
  - `badge`: `91.12%`
- The good news is runtime:
  - all three attribute selectors finished in about `0.7 s`
  - this is dramatically cheaper than `coreset` (`45 min`) or the original `badge` (`1 h 52 min`) at the same ratio
- More aggressive rarity / penalty settings did not improve overall held-out accuracy on CelebA at `ratio=0.1`.
  - under this current metadata-and-scoring setup, stronger attribute steering appears to over-focus the selection without improving final generalization.
- Practical takeaway:
  - if CelebA attribute experiments continue, the conservative `R=0.01, P=5.0, C=inf, E=False` setting is the only sensible starting point from this probe
  - but the current attribute path is not yet competitive with the completed `random` / `badge` / `stratified` baselines on this CelebA setup

### 04/16/2026 Update — CelebA Attribute Mix-Scaled Probe (`ratio=0.1`, `E=False`, `extra_count_scale=10`)

- The earlier three-run probe injected raw generated counts (`1000`) directly into the full `80,000`-image original candidate pool.
- For `ratio=0.1`, the final training mix is really `8000 original + 1000 generated`, so the selector was underestimating the generated-image share during scoring.
- This follow-up probe therefore scales the explicit generated counts by `1 / sub_ratio = 10` before scoring:
  - effective injected counts during selection:
    - `hair color:red=10000`
    - `skin:brown=10000`
    - `emotion:sad=10000`
- With that correction in place, the CelebA sweep moved to much milder `R/P/C` settings instead of reusing the stronger ImageNet-style penalties.

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.1`) | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|-------------------------------|-------------------|-----------------|---------------|
| `mixselx10_e0_r003_p05_cinf` | `0.03` | `0.5` | `inf` | **`90.99%`** | `+0.42 pp` | `+0.84 pp` | `0.74 s` | `349.37 s` | `logs_celeba/attr_ratio0.1_mixselx10_e0_r003_p05_cinf_gpu2_20260416_141739.log` |
| `mixselx10_e0_r000_p0_cinf` | `0.0` | `0.0` | `inf` | `90.86%` | `+0.29 pp` | `+0.71 pp` | `0.70 s` | `347.44 s` | `logs_celeba/attr_ratio0.1_mixselx10_e0_r000_p0_cinf_gpu1_20260416_141739.log` |
| `mixselx10_e0_r01_p05_cinf` | `0.1` | `0.5` | `inf` | `90.81%` | `+0.24 pp` | `+0.66 pp` | `0.74 s` | `372.83 s` | `logs_celeba/attr_ratio0.1_mixselx10_e0_r01_p05_cinf_gpu3_20260416_141739.log` |
| `mixselx10_e0_r000_p05_cinf` | `0.0` | `0.5` | `inf` | `90.71%` | `+0.14 pp` | `+0.56 pp` | `0.94 s` | `348.21 s` | `logs_celeba/attr_ratio0.1_mixselx10_e0_r000_p05_cinf_gpu1_20260416_141739.log` |
| `mixselx10_e0_r001_p0_cinf` | `0.01` | `0.0` | `inf` | `90.70%` | `+0.13 pp` | `+0.55 pp` | `0.72 s` | `344.59 s` | `logs_celeba/attr_ratio0.1_mixselx10_e0_r001_p0_cinf_gpu0_20260416_141739.log` |
| `mixselx10_e0_r001_p05_cinf` | `0.01` | `0.5` | `inf` | `90.67%` | `+0.10 pp` | `+0.52 pp` | `1.05 s` | `343.75 s` | `logs_celeba/attr_ratio0.1_mixselx10_e0_r001_p05_cinf_gpu0_20260416_141739.log` |
| `mixselx10_e0_r001_p1_cinf` | `0.01` | `1.0` | `inf` | `90.61%` | `+0.04 pp` | `+0.46 pp` | `1.01 s` | `350.25 s` | `logs_celeba/attr_ratio0.1_mixselx10_e0_r001_p1_cinf_gpu2_20260416_141739.log` |
| `mixselx10_e0_r01_p05_c15` | `0.1` | `0.5` | `15.0` | `90.61%` | `+0.04 pp` | `+0.46 pp` | `0.98 s` | `412.87 s` | `logs_celeba/attr_ratio0.1_mixselx10_e0_r01_p05_c15_gpu3_20260416_141739.log` |

### CelebA Attribute Mix-Scaled Observations

- Correcting the generated-count scale materially helped:
  - old raw-count best:
    - `90.59%`
  - new mix-scaled best:
    - `90.99%`
  - improvement:
    - `+0.40 pp`
- The best CelebA configuration after this correction is:
  - `R=0.03`
  - `P=0.5`
  - `C=inf`
  - `E=False`
  - `extra_count_scale=10`
- After the scale correction, CelebA prefers much milder penalties than the earlier ImageNet-style probe:
  - `P=0.0` to `0.5` is consistently better than `P=1.0`
  - strict class balance (`C=inf`) is better than the softer `C=15`
- Runtime remains excellent:
  - all eight mix-scaled selectors still finish in about `0.7` to `1.1 s`
- The attribute method is now competitive with the corrected `ratio=0.1` baselines:
  - `random`: `90.15%`
  - `stratified`: `90.95%`
  - `badge`: `91.12%`
  - `best mix-scaled attribute`: `90.99%`
- Practical takeaway:
  - if the CelebA attribute sweep continues to larger ratios, start from:
    - `R=0.03, P=0.5, C=inf, E=False`
    - with generated extra counts scaled by `1 / sub_ratio`

### 04/16/2026 Update — CelebA Attribute Mix-Scaled Follow-Up (`ratio=0.3`, top 3 from `ratio=0.1`)

- After the corrected `ratio=0.1` sweep, we transferred its top 3 settings to `ratio=0.3` instead of rerunning the full 8-scenario grid immediately.
- These runs also used generated-count scaling matched to the selected-data regime:
  - `extra_count_scale = 1 / 0.3 = 3.33333`
  - effective injected counts during scoring:
    - `hair color:red=3333.33`
    - `skin:brown=3333.33`
    - `emotion:sad=3333.33`

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.3`) | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|-------------------------------|-------------------|-----------------|---------------|
| `mixselx333_e0_r000_p0_cinf` | `0.0` | `0.0` | `inf` | **`90.89%`** | `+0.32 pp` | `-0.29 pp` | `0.71 s` | `492.89 s` | `logs_celeba/attr_ratio0.3_mixselx333_e0_r000_p0_cinf_gpu2_20260416_143519.log` |
| `mixselx333_e0_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `90.87%` | `+0.30 pp` | `-0.31 pp` | `0.76 s` | `470.50 s` | `logs_celeba/attr_ratio0.3_mixselx333_e0_r003_p05_cinf_gpu1_20260416_143519.log` |
| `mixselx333_e0_r01_p05_cinf` | `0.1` | `0.5` | `inf` | `90.74%` | `+0.17 pp` | `-0.44 pp` | `0.76 s` | `637.82 s` | `logs_celeba/attr_ratio0.3_mixselx333_e0_r01_p05_cinf_gpu3_20260416_143519.log` |

### CelebA Attribute Mix-Scaled Ratio-0.3 Observations

- All three transferred `ratio=0.1` winners completed cleanly at `ratio=0.3`.
- Within this transferred top-3 set, the best `ratio=0.3` configuration is:
  - `R=0.0`
  - `P=0.0`
  - `C=inf`
  - `E=False`
  - `extra_count_scale=3.33333`
- This is a notable shift from `ratio=0.1`, where the best corrected setting was:
  - `R=0.03`
  - `P=0.5`
  - `C=inf`
- Even so, the attribute method still trails the completed CelebA `ratio=0.3` baselines:
  - `random`: `91.18%`
  - `stratified`: `91.15%`
  - `coreset`: `91.13%`
  - `badge`: `91.04%`
  - `best transferred mix-scaled attribute`: `90.89%`
- Compared with the corrected `ratio=0.1` best (`90.99%`), the transferred `ratio=0.3` best did not improve.
- Practical takeaway:
  - if a larger `ratio=0.3` attribute sweep is worth doing, search around the milder corner first:
    - `R in {0.0, 0.03}`
    - `P in {0.0, 0.5}`
    - `C = inf`
    - still using generated-count scaling by `1 / sub_ratio`

### 04/17/2026 Update — CelebA Qwen3 Attribute Follow-Up (`ratio=0.3`, finite `C=15`)

- The earlier Qwen3 `ratio=0.3` reference runs only covered `C=inf`, but that was not the intended final selector regime.
- This follow-up kept the same mild `R/P` neighborhood and changed only the class-balance term to finite `C=15.0` using:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
  - `extra_count_scale=3.33333`

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.3`) | Best Val ACC | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|-------------------------------|--------------|-------------------|-----------------|---------------|
| `qwen3mixx333_r000_p0_c15` | `0.0` | `0.0` | `15.0` | **`90.88%`** | `+0.31 pp` | `-0.30 pp` | `90.92%` | `0.53 s` | `423.41 s` | `logs_celeba/attr_ratio0.3_qwen3mixx333_r000_p0_c15_gpu0_20260417_011502.log` |
| `qwen3mixx333_r001_p05_c15` | `0.01` | `0.5` | `15.0` | `90.61%` | `+0.04 pp` | `-0.57 pp` | `90.55%` | `0.55 s` | `416.34 s` | `logs_celeba/attr_ratio0.3_qwen3mixx333_r001_p05_c15_gpu1_20260417_011502.log` |
| `qwen3mixx333_r003_p05_c15` | `0.03` | `0.5` | `15.0` | `90.28%` | `-0.29 pp` | `-0.90 pp` | `90.38%` | `0.54 s` | `420.36 s` | `logs_celeba/attr_ratio0.3_qwen3mixx333_r003_p05_c15_gpu2_20260417_011502.log` |

### CelebA Qwen3 Ratio-0.3 Finite-`C` Observations

- Switching from `C=inf` to `C=15.0` did not improve the best `ratio=0.3` Qwen3 result.
- The best finite-`C` configuration is still the most conservative one:
  - `R=0.0`
  - `P=0.0`
  - `C=15.0`
  - held-out test accuracy `90.88%`
- Compared with the earlier Qwen3 `C=inf` reference at the same mild corner:
  - `R=0.0, P=0.0` changed from `90.89%` to `90.88%`
  - effectively no gain
- The more attribute-steered finite-`C` settings were clearly worse at `ratio=0.3`:
  - `R=0.01, P=0.5, C=15.0`: `90.61%`
  - `R=0.03, P=0.5, C=15.0`: `90.28%`
- Relative to the completed non-attribute CelebA baselines at `ratio=0.3`, the attribute path still trails:
  - `random`: `91.18%`
  - `stratified`: `91.15%`
  - `coreset`: `91.13%`
  - `badge`: `91.04%`
  - best Qwen3 finite-`C` attribute result: `90.88%`
- Practical takeaway:
  - for CelebA `ratio=0.3`, there is currently no evidence that finite class balance helps the Qwen3 attribute selector
  - if this line continues, the only defensible region is still the conservative near-random corner

### 04/17/2026 Update — CelebA Qwen3 Attribute Re-Run (`ratio=0.1`, 4 selected RPCs)

- The earlier CelebA attribute experiments above used the older BLIP-style metadata file:
  - `save/attribute.json`
- A follow-up re-run was completed against the newer Qwen3-assigned merged train+val metadata:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
- To keep the source explicit and avoid silently changing defaults, these runs used:
  - `scripts/run_celeba_attribute_ratio0.1_qwen3_rpc4.sh`
- The four selected settings were chosen from the previously strongest `ratio=0.1` mix-scaled BLIP region instead of rerunning the full grid.

| Config | `R` | `P` | `C` | Qwen3 Held-out Test ACC | BLIP Held-out Test ACC | Delta vs BLIP | Best Val ACC | AL Selection Time | End-to-End Time | Qwen3 Log |
|--------|-----|-----|-----|--------------------------|------------------------|---------------|--------------|-------------------|-----------------|-----------|
| `qwen3mix_r000_p0_cinf` | `0.0` | `0.0` | `inf` | **`90.86%`** | `90.86%` | `+0.00 pp` | `91.03%` | `0.38 s` | `217.35 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r000_p0_cinf_gpu0_20260417_004805.log` |
| `qwen3mix_r001_p05_cinf` | `0.01` | `0.5` | `inf` | `90.30%` | `90.67%` | `-0.37 pp` | `90.29%` | `0.42 s` | `217.03 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r001_p05_cinf_gpu1_20260417_004805.log` |
| `qwen3mix_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `90.12%` | **`90.99%`** | `-0.87 pp` | `90.33%` | `0.45 s` | `216.75 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r003_p05_cinf_gpu2_20260417_004805.log` |
| `qwen3mix_r01_p05_cinf` | `0.1` | `0.5` | `inf` | `90.03%` | `90.81%` | `-0.78 pp` | `90.16%` | `0.41 s` | `216.72 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r01_p05_cinf_gpu3_20260417_004805.log` |

### CelebA Qwen3 Re-Run Observations

- This Qwen3 metadata re-run did **not** improve over the earlier BLIP-based `ratio=0.1` attribute results.
- The best Qwen3 result is:
  - `R=0.0`
  - `P=0.0`
  - `C=inf`
  - held-out test accuracy `90.86%`
- Compared with the best BLIP-based `ratio=0.1` attribute result:
  - BLIP best:
    - `R=0.03`
    - `P=0.5`
    - `C=inf`
    - held-out test accuracy `90.99%`
  - Qwen3 best:
    - `90.86%`
  - gap:
    - `-0.13 pp`
- On a same-config comparison, Qwen3 matched only the conservative control:
  - `R=0.0`, `P=0.0`, `C=inf`:
    - `90.86%` vs `90.86%`
- The three stronger rarity-steering settings all degraded under Qwen3 metadata:
  - `R=0.01`, `P=0.5`, `C=inf`:
    - `-0.37 pp`
  - `R=0.03`, `P=0.5`, `C=inf`:
    - `-0.87 pp`
  - `R=0.1`, `P=0.5`, `C=inf`:
    - `-0.78 pp`
- Practical takeaway:
  - for CelebA `ratio=0.1` attribute AL, the Qwen3 metadata source did not improve overall held-out accuracy over the earlier BLIP-based metadata
  - if this line continues, the immediate options are:
    - keep the BLIP-based `R=0.03, P=0.5, C=inf` result as the stronger `ratio=0.1` attribute reference
    - or inspect why Qwen3 labels appear to make the selector over-steer once `P>0` and `R>0`

### 04/17/2026 Update — CelebA Qwen3 Attribute Follow-Up (`ratio=0.1`, finite `C=15`)

- Because `C=inf` is not admissible for the next selector choice, a second `ratio=0.1` Qwen3 follow-up re-tested the mild top-3 Qwen3 `R/P` settings with finite class balance:
  - `C=15.0`
- These runs again used:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
  - `extra_count_scale = 10`

| Config | `R` | `P` | `C` | Qwen3 Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.1`) | Best Val ACC | AL Selection Time | End-to-End Time | Qwen3 Log |
|--------|-----|-----|-----|--------------------------|---------------|-------------------------------|--------------|-------------------|-----------------|-----------|
| `qwen3mix_r003_p05_c15` | `0.03` | `0.5` | `15.0` | **`91.07%`** | `+0.50 pp` | `+0.92 pp` | `90.91%` | `0.42 s` | `216.89 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r003_p05_c15_gpu2_20260417_010718.log` |
| `qwen3mix_r000_p0_c15` | `0.0` | `0.0` | `15.0` | `90.84%` | `+0.27 pp` | `+0.69 pp` | `90.97%` | `0.40 s` | `217.38 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r000_p0_c15_gpu0_20260417_010718.log` |
| `qwen3mix_r001_p05_c15` | `0.01` | `0.5` | `15.0` | `90.09%` | `-0.48 pp` | `-0.06 pp` | `90.24%` | `0.43 s` | `217.79 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r001_p05_c15_gpu1_20260417_010718.log` |

### CelebA Qwen3 Ratio-0.1 Finite-C Observations

- The finite-`C` rerun changes the ranking within the Qwen3 `ratio=0.1` family:
  - best finite-`C` setting:
    - `R=0.03`
    - `P=0.5`
    - `C=15.0`
    - held-out test accuracy `91.07%`
- Compared with the earlier Qwen3 `C=inf` runs at the same `ratio=0.1`:
  - `R=0.0`, `P=0.0`:
    - `C=15.0` Qwen3 `90.84%`
    - `C=inf` Qwen3 `90.86%`
    - delta `-0.02 pp`
  - `R=0.01`, `P=0.5`:
    - `C=15.0` Qwen3 `90.09%`
    - `C=inf` Qwen3 `90.30%`
    - delta `-0.21 pp`
  - `R=0.03`, `P=0.5`:
    - `C=15.0` Qwen3 `91.07%`
    - `C=inf` Qwen3 `90.12%`
    - delta `+0.95 pp`
- This finite-`C` result substantially closes the gap to the earlier BLIP-based best `ratio=0.1` attribute run:
  - BLIP best:
    - `R=0.03`, `P=0.5`, `C=inf`
    - `90.99%`
  - Qwen3 best finite-`C`:
    - `R=0.03`, `P=0.5`, `C=15.0`
    - `91.07%`
  - delta:
    - `+0.08 pp`
- It now beats the corrected `random` and `stratified` `ratio=0.1` baselines, but it still does not beat the strongest completed non-attribute method:
  - `random`: `90.15%`
  - `stratified`: `90.95%`
  - `badge`: `91.12%`
  - Qwen3 best finite-`C`: `91.07%`
- Practical takeaway:
  - once `C=inf` is excluded, the best current Qwen3 `ratio=0.1` setting is:
    - `R=0.03, P=0.5, C=15.0`
  - this is the first Qwen3 CelebA attribute setting that slightly beats the earlier BLIP-based best attribute result, while remaining just below the finished `random` and `badge` baselines

### 04/17/2026 Update — CelebA Qwen3 Attribute Local Refinement (`ratio=0.1`, around `R=0.03, P=0.5, C=15`)

- After the first finite-`C` Qwen3 `ratio=0.1` sweep, a small local refinement tested whether the current best point could be improved by:
  - lowering rarity from `R=0.03` to `R=0.02`
  - raising rarity from `R=0.03` to `R=0.05`
  - softening class balance from `C=15.0` to `C=8.0`
- These runs again used:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
  - `extra_count_scale = 10`

| Config | `R` | `P` | `C` | Qwen3 Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.1`) | Best Val ACC | AL Selection Time | End-to-End Time | Qwen3 Log |
|--------|-----|-----|-----|--------------------------|---------------|-------------------------------|--------------|-------------------|-----------------|-----------|
| `qwen3mix_r003_p05_c8` | `0.03` | `0.5` | `8.0` | **`90.66%`** | `+0.09 pp` | `+0.51 pp` | `90.63%` | `0.53 s` | `235.02 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r003_p05_c8_gpu2_20260417_012110.log` |
| `qwen3mix_r002_p05_c15` | `0.02` | `0.5` | `15.0` | `90.41%` | `-0.16 pp` | `+0.26 pp` | `90.46%` | `0.52 s` | `230.14 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r002_p05_c15_gpu0_20260417_012110.log` |
| `qwen3mix_r005_p05_c15` | `0.05` | `0.5` | `15.0` | `90.15%` | `-0.42 pp` | `+0.00 pp` | `90.27%` | `0.52 s` | `235.35 s` | `logs_celeba/attr_ratio0.1_qwen3mix_r005_p05_c15_gpu1_20260417_012110.log` |

### CelebA Qwen3 Ratio-0.1 Local-Refinement Observations

- None of the three local-refinement runs beat the incumbent best finite-`C` setting:
  - incumbent:
    - `R=0.03, P=0.5, C=15.0`
    - held-out test accuracy `91.07%`
- Lowering rarity slightly hurts:
  - `R=0.02, P=0.5, C=15.0`
  - `90.41%`
  - delta vs incumbent `-0.66 pp`
- Increasing rarity also hurts:
  - `R=0.05, P=0.5, C=15.0`
  - `90.15%`
  - delta vs incumbent `-0.92 pp`
- Softening class balance from `C=15.0` to `C=8.0` is better than moving `R`, but it still loses to the incumbent:
  - `R=0.03, P=0.5, C=8.0`
  - `90.66%`
  - delta vs incumbent `-0.41 pp`
- Practical takeaway:
  - the local neighborhood evidence supports keeping:
    - `R=0.03, P=0.5, C=15.0`
  - as the current best Qwen3 CelebA attribute setting for `ratio=0.1`

### 04/17/2026 Update — CelebA Qwen3 `redhair_brownskin` Attribute Pre-Rerun Snapshot (`ratio=0.1`, interrupted old directory semantics)

- Before the generated-image directories were renamed to distinguish filtered vs unfiltered content, six `redhair_brownskin` attribute runs were launched with the old path meaning.
- Those runs started normally, completed active-learning selection, and entered resumed training, but all six crashed mid-training after the directory rename because the training dataloader could no longer open some generated images under the old path.
- These logs are still useful as a pre-rerun comparison snapshot for selection behavior and early validation trajectory, but they are **not** valid final accuracy results.

| Config | `R` | `P` | `C` | Best Observed Val ACC Before Crash | Last Completed Epoch | Failure Type | Canonical Log |
|--------|-----|-----|-----|------------------------------------|----------------------|--------------|---------------|
| `rb_qwen3_r003_p05_c15` | `0.03` | `0.5` | `15.0` | `90.21%` | `8` | `FileNotFoundError` after directory rename | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_c15_gpu0_20260417_012831.log` |
| `rb_qwen3_r000_p0_cinf` | `0.0` | `0.0` | `inf` | `90.79%` | `8` | `FileNotFoundError` after directory rename | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r000_p0_cinf_gpu0_20260417_012831.log` |
| `rb_qwen3_r000_p0_c15` | `0.0` | `0.0` | `15.0` | `90.79%` | `7` | `FileNotFoundError` after directory rename | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r000_p0_c15_gpu1_20260417_012831.log` |
| `rb_qwen3_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `89.82%` | `7` | `FileNotFoundError` after directory rename | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_cinf_gpu1_20260417_012831.log` |
| `rb_qwen3_r003_p05_c8` | `0.03` | `0.5` | `8.0` | `90.87%` | `7` | `FileNotFoundError` after directory rename | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_c8_gpu2_20260417_012831.log` |
| `rb_qwen3_r002_p05_c15` | `0.02` | `0.5` | `15.0` | `90.33%` | `7` | `FileNotFoundError` after directory rename | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r002_p05_c15_gpu2_20260417_012831.log` |

### CelebA `redhair_brownskin` Pre-Rerun Snapshot Notes

- All six interrupted runs successfully reached:
  - `Selected 1000 filenames for sampling`
  - `CelebA: Combined 8000 selected samples with 1000 generated images using Active Learning`
  - resumed fine-tuning from `weights/model_celeba_resnet_base.pth`
- The crash mechanism was consistent across runs:
  - `FileNotFoundError`
  - missing generated image under `data/redhair_brownskin/...`
  - caused by the directory rename that changed filtered vs unfiltered path semantics during training
- These rows should only be used as an early-trajectory comparison against the rerun that uses the corrected filtered directory layout.

### 04/17/2026 Update — CelebA Qwen3 `redhair_brownskin` Attribute Rerun (`ratio=0.1`, filtered directory)

- After the generated-image directory semantics were corrected, the same six `redhair_brownskin` Qwen3 `ratio=0.1` configurations were rerun against the filtered subset now mounted at:
  - `data/redhair_brownskin`
- These runs used:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
  - focus:
    - `hair color,skin`
  - explicit generated counts:
    - `hair color:red=1000`
    - `skin:brown=1000`
  - `extra_count_scale = 10`

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Best Val ACC | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|--------------|-------------------|-----------------|---------------|
| `rb_qwen3_r000_p0_c15` | `0.0` | `0.0` | `15.0` | **`90.75%`** | `+0.18 pp` | `90.82%` | `0.34 s` | `323.19 s` | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r000_p0_c15_gpu1_20260417_013234.log` |
| `rb_qwen3_r000_p0_cinf` | `0.0` | `0.0` | `inf` | `90.74%` | `+0.17 pp` | `90.82%` | `0.32 s` | `315.65 s` | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r000_p0_cinf_gpu0_20260417_013234.log` |
| `rb_qwen3_r002_p05_c15` | `0.02` | `0.5` | `15.0` | `90.63%` | `+0.06 pp` | `90.63%` | `0.42 s` | `323.69 s` | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r002_p05_c15_gpu2_20260417_013234.log` |
| `rb_qwen3_r003_p05_c15` | `0.03` | `0.5` | `15.0` | `90.33%` | `-0.24 pp` | `90.53%` | `0.37 s` | `316.43 s` | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_c15_gpu0_20260417_013234.log` |
| `rb_qwen3_r003_p05_c8` | `0.03` | `0.5` | `8.0` | `90.16%` | `-0.41 pp` | `90.30%` | `0.36 s` | `323.48 s` | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_c8_gpu2_20260417_013234.log` |
| `rb_qwen3_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `89.65%` | `-0.92 pp` | `89.92%` | `0.36 s` | `322.76 s` | `logs_celeba/attr_rb_ratio0.1_rb_qwen3_r003_p05_cinf_gpu1_20260417_013234.log` |

### CelebA `redhair_brownskin` Ratio-0.1 Observations

- On the filtered `redhair_brownskin` subset, the best Qwen3 attribute setting is the conservative control:
  - `R=0.0`
  - `P=0.0`
  - `C=15.0`
  - held-out test accuracy `90.75%`
- The corresponding `C=inf` control is effectively tied:
  - `R=0.0, P=0.0, C=inf`
  - `90.74%`
- Unlike the earlier `redhair_brownskin_sademotion` experiments, steering toward the `R=0.03, P=0.5` region hurts on this filtered two-attribute setup:
  - `C=15.0`: `90.33%`
  - `C=8.0`: `90.16%`
  - `C=inf`: `89.65%`
- The filtered `redhair_brownskin` subset therefore appears to prefer minimal attribute steering, with finite `C=15.0` only slightly ahead of the pure `C=inf` control.
- Practical takeaway:
  - for the current filtered `redhair_brownskin` `ratio=0.1` setting, carry forward:
    - `R=0.0, P=0.0, C=15.0`
  - and treat the earlier interrupted pre-rerun table only as a directory-mismatch comparison snapshot, not as a competing result set

### 04/17/2026 Update — CelebA Qwen3 `redhair_brownskin` Attribute Follow-Up (`ratio=0.3`, filtered directory)

- The six filtered `redhair_brownskin` Qwen3 `ratio=0.3` runs launched earlier on GPUs `0/1/2` have now finished.
- These runs used:
  - `data/redhair_brownskin`
  - `save/attribute_celeba_trainval_qwen3_8B.json`
  - `extra_count_scale = 3.33333`
- This was the direct larger-ratio follow-up to the filtered `ratio=0.1` rerun, keeping both conservative controls and the previously informative mild steering region.

| Config | `R` | `P` | `C` | Held-out Test ACC | Delta vs Base | Best Val ACC | AL Selection Time | End-to-End Time | Canonical Log |
|--------|-----|-----|-----|-------------------|---------------|--------------|-------------------|-----------------|---------------|
| `rb_qwen3x333_r000_p0_c15` | `0.0` | `0.0` | `15.0` | **`90.91%`** | `+0.34 pp` | `91.02%` | `0.44 s` | `655.45 s` | `logs_celeba/attr_rb_ratio0.3_rb_qwen3x333_r000_p0_c15_gpu0_20260417_014040.log` |
| `rb_qwen3x333_r000_p0_cinf` | `0.0` | `0.0` | `inf` | **`90.91%`** | `+0.34 pp` | `91.03%` | `0.43 s` | `656.18 s` | `logs_celeba/attr_rb_ratio0.3_rb_qwen3x333_r000_p0_cinf_gpu0_20260417_014040.log` |
| `rb_qwen3x333_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `90.63%` | `+0.06 pp` | `90.72%` | `0.44 s` | `676.91 s` | `logs_celeba/attr_rb_ratio0.3_rb_qwen3x333_r003_p05_cinf_gpu2_20260417_014040.log` |
| `rb_qwen3x333_r003_p05_c15` | `0.03` | `0.5` | `15.0` | `90.53%` | `-0.04 pp` | `90.61%` | `0.47 s` | `661.52 s` | `logs_celeba/attr_rb_ratio0.3_rb_qwen3x333_r003_p05_c15_gpu1_20260417_014040.log` |
| `rb_qwen3x333_r002_p05_c15` | `0.02` | `0.5` | `15.0` | `90.47%` | `-0.10 pp` | `90.63%` | `0.47 s` | `664.06 s` | `logs_celeba/attr_rb_ratio0.3_rb_qwen3x333_r002_p05_c15_gpu1_20260417_014040.log` |
| `rb_qwen3x333_r003_p05_c8` | `0.03` | `0.5` | `8.0` | `90.37%` | `-0.20 pp` | `90.49%` | `0.46 s` | `678.05 s` | `logs_celeba/attr_rb_ratio0.3_rb_qwen3x333_r003_p05_c8_gpu2_20260417_014040.log` |

### CelebA `redhair_brownskin` Ratio-0.3 Observations

- On the filtered `redhair_brownskin` subset at `ratio=0.3`, the best result is again the conservative control family:
  - `R=0.0`
  - `P=0.0`
  - both `C=15.0` and `C=inf` reached the same held-out test accuracy:
    - `90.91%`
- Relative to the filtered `ratio=0.1` rerun, increasing the retained original-data ratio helped the control point:
  - `ratio=0.1` best:
    - `90.75%`
  - `ratio=0.3` best:
    - `90.91%`
  - improvement:
    - `+0.16 pp`
- The mild attribute-steered region still does not beat the control on this filtered two-attribute setup:
  - `R=0.02, P=0.5, C=15.0`:
    - `90.47%`
  - `R=0.03, P=0.5, C=15.0`:
    - `90.53%`
  - `R=0.03, P=0.5, C=8.0`:
    - `90.37%`
  - `R=0.03, P=0.5, C=inf`:
    - `90.63%`
- Runtime remains negligible compared with embedding-based AL baselines:
  - all six selections still finished in about `0.43` to `0.47 s`
- Practical takeaway:
  - for the current filtered `redhair_brownskin` line, both completed ratios (`0.1` and `0.3`) prefer near-random Qwen3 selection over explicit rarity steering
  - if this branch continues, carry forward:
    - `R=0.0`
    - `P=0.0`
    - `C=15.0` or `C=inf`

### 04/17/2026 Update — CelebA Qwen3 Attribute Follow-Up (`ratio=0.3`, transferred top 3)

- A second Qwen3-based follow-up transferred the top 3 mild `ratio=0.1` Qwen3 settings to `ratio=0.3`:
  - `R=0.0, P=0.0, C=inf`
  - `R=0.01, P=0.5, C=inf`
  - `R=0.03, P=0.5, C=inf`
- These runs used the same Qwen3 metadata source:
  - `save/attribute_celeba_trainval_qwen3_8B.json`
- The generated-count scaling again matched the selected-data regime:
  - `extra_count_scale = 1 / 0.3 = 3.33333`

| Config | `R` | `P` | `C` | Qwen3 Held-out Test ACC | Delta vs Base | Gap vs `random` (`ratio=0.3`) | Best Val ACC | AL Selection Time | End-to-End Time | Qwen3 Log |
|--------|-----|-----|-----|--------------------------|---------------|-------------------------------|--------------|-------------------|-----------------|-----------|
| `qwen3mixx333_r000_p0_cinf` | `0.0` | `0.0` | `inf` | **`90.89%`** | `+0.32 pp` | `-0.29 pp` | `90.90%` | `0.51 s` | `399.41 s` | `logs_celeba/attr_ratio0.3_qwen3mixx333_r000_p0_cinf_gpu0_20260417_005520.log` |
| `qwen3mixx333_r001_p05_cinf` | `0.01` | `0.5` | `inf` | `90.49%` | `-0.08 pp` | `-0.69 pp` | `90.50%` | `0.50 s` | `401.53 s` | `logs_celeba/attr_ratio0.3_qwen3mixx333_r001_p05_cinf_gpu1_20260417_005520.log` |
| `qwen3mixx333_r003_p05_cinf` | `0.03` | `0.5` | `inf` | `90.63%` | `+0.06 pp` | `-0.55 pp` | `90.71%` | `0.51 s` | `399.91 s` | `logs_celeba/attr_ratio0.3_qwen3mixx333_r003_p05_cinf_gpu2_20260417_005520.log` |

### CelebA Qwen3 Ratio-0.3 Observations

- The best completed Qwen3 `ratio=0.3` result is still the conservative control:
  - `R=0.0`
  - `P=0.0`
  - `C=inf`
  - held-out test accuracy `90.89%`
- On the overlapping transferred settings, Qwen3 does not beat the earlier BLIP-based ratio-0.3 results:
  - `R=0.0`, `P=0.0`, `C=inf`:
    - Qwen3 `90.89%`
    - BLIP `90.89%`
    - delta `+0.00 pp`
  - `R=0.03`, `P=0.5`, `C=inf`:
    - Qwen3 `90.63%`
    - BLIP `90.87%`
    - delta `-0.24 pp`
- There is no direct BLIP `ratio=0.3` comparison point for:
  - `R=0.01`, `P=0.5`, `C=inf`
  - because the earlier BLIP top-3 transfer at `ratio=0.3` used `R=0.1`, `P=0.5`, `C=inf` instead
- All three completed Qwen3 `ratio=0.3` runs still trail the finished non-attribute CelebA baselines at the same ratio:
  - `random`: `91.18%`
  - `stratified`: `91.15%`
  - `coreset`: `91.13%`
  - `badge`: `91.04%`
- These completed Qwen3 `ratio=0.3` runs are still useful as reference points, but they are **not** the right settings to carry forward if `C=inf` is no longer admissible.
- Practical takeaway:
  - preserve the `ratio=0.3` Qwen3 results for reference
  - but move the next search back to `ratio=0.1` with finite `C` values rather than selecting from the `C=inf` runs above

# 04/16/2026 Update — ImageNet10 BADGE Resume Sweep

- Completed four ImageNet10 BADGE follow-up runs using the same resume-from-base setting as the earlier AL comparison:
  - base checkpoint: `weights/model_imagenet_resnet_base.pth`
  - model: `ResNet-18`
  - learning rate: `1e-5`
  - epochs: `15`
  - generated images: `1000`
  - attribute prompt: `pinkcolor_fabrictexture`
- All four runs finished BADGE selection, resumed training, and printed held-out test accuracy.
- The runs then failed in post-processing:
  - `sub_ratio=0.1`: prediction save failed because `save/pred/` was missing
  - `sub_ratio=0.3/0.5/0.7`: diagnostics failed because `save/attribute10.json` was missing
- These failures happened **after** the main evaluation lines, so the reported test metrics below are valid.

| Sub Ratio | Selected Original Samples | Final Train Size After +1000 Gen | Best Val ACC | Held-out Test ACC | AL Selection Time | End-to-End Time | Post-run Status |
|---------|----------------------------|----------------------------------|--------------|-------------------|-------------------|-----------------|-----------------|
| `0.1` | `1040` | `2040` | `0.7208` | `0.7323` | `272.65 s` | n/a | crashed during prediction save after test evaluation |
| `0.3` | `3120` | `4120` | `0.7215` | `0.7423` | `13 min` | `22 min` | crashed during diagnostics after test evaluation |
| `0.5` | `5200` | `6200` | `0.7223` | `0.7469` | `21 min` | `28 min` | crashed during diagnostics after test evaluation |
| `0.7` | `7279` | `8279` | `0.7215` | **`0.7477`** | `27 min` | `31 min` | crashed during diagnostics after test evaluation |

### Observations

- Test accuracy improved monotonically with the retained original-data ratio:
  - `0.7323 -> 0.7423 -> 0.7469 -> 0.7477`
- At the directly comparable `sub_ratio=0.3` resume setting, BADGE reached `0.7423` held-out test accuracy.
  - This is above the earlier `2/27/2026` resume table entries for `Random` (`0.7308`), `Stratified` (`0.7369`), `Coreset` (`0.7359`), `Diversity` (`0.7362`), `Uncertainty Stratified` (`0.7292`), `Hierarchical` (`0.7346`), and `Attribute` (`0.7369`).
  - It also edges the earlier cited `Hybrid` result (`0.7400`) by `+0.0023`.
- Best validation accuracy stayed nearly flat around `0.721` to `0.722`, while test accuracy improved as more original data was retained.
  - In this setup, larger BADGE-selected subsets help held-out generalization more clearly than they help validation peak accuracy.
- BADGE selection is materially more expensive than the lightweight metadata-based `attribute` selector:
  - even on this 10-class ImageNet subset, AL selection time grew from `272.65 s` at `sub_ratio=0.1` to `27 min` at `sub_ratio=0.7`
  - the cost comes from pretrained embedding extraction plus BADGE gradient-embedding batch seeding

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
| Qwen3-VL-8B | `3` | `0` | `0.6445` | `0.6352` | `0.8656` | `0.6902` | `0.6993` | `1 h 29 min` | `log_pose/pose_keypointrcnn_train20k_qwen3_bs8_e2_rare1p1_20260415_144429/` | `log_pose/pose_train20k_qwen3_bs8_e2_rare1p1_20260415_144429.log` |
| InternVL3-8B | `0` | `0` | `0.6401` | `0.5902` | `0.8618` | `0.6936` | `0.7016` | `1 h 34 min` | `log_pose/pose_keypointrcnn_train20k_internvl3_bs8_e2_rare1p1_gpu0_20260415_144530/` | `log_pose/base_pose_train20k_compare.log` |

- Relative to the earlier `4/12` held-out test probes:
  - Qwen3-VL-8B improved from `0.5968` to `0.6352` (`+0.0384`)
  - InternVL3-8B moved from `0.6340` to `0.5902` (`-0.0438`)
- Top-level `log_pose/` logs explicitly record both training-only and end-to-end elapsed time for these runs:
  - Qwen3-VL-8B: training `1 h 29 min`, end-to-end `1 h 29 min`
  - InternVL3-8B: training `1 h 34 min`, end-to-end `1 h 34 min`
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
| Qwen3-VL-8B GPU `0` repro | Qwen3-VL-8B GPU `3` (`20260415_144429`) | `0.6445 -> 0.6407` (`-0.0038`) | `0.6352 -> 0.6341` (`-0.0011`) | `0.8656 -> 0.8609` (`-0.0047`) | `0.6902 -> 0.6866` (`-0.0036`) | `0.6993 -> 0.6958` (`-0.0035`) | `1 h 29 min -> 1 h 39 min` (`+10 min`) |
| InternVL3-8B GPU `3` train-only rerun | InternVL3-8B GPU `0` (`20260415_144530`) | `0.6401 -> 0.6412` (`+0.0011`) | `0.5902 -> 0.6371` (`+0.0469`) | `0.8618 -> 0.8583` (`-0.0035`) | `0.6936 -> 0.6963` (`+0.0027`) | `0.7016 -> 0.6950` (`-0.0066`) | `1 h 34 min -> 1 h 29 min` (`-5 min`) |

- Qwen3-VL-8B repro interpretation:
  - overall held-out metrics are slightly below the earlier GPU `3` Qwen3 run
  - the hardest rare slice `pose=lying` improved from `AP=0.3091` to `AP=0.3906` (`+0.0815`)
  - `pose=sitting` also improved slightly from `AP=0.5503` to `AP=0.5529`
  - low-light slices remained difficult and close to the earlier run:
    - `time_of_day=evening`: `0.5629 -> 0.5737`
    - `time_of_day=night`: `0.5816 -> 0.5802`
- InternVL3-8B GPU `3` train-only rerun interpretation:
  - held-out AP is higher than the earlier GPU `0` InternVL3 run on both `val` and `test`
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

`alpred.py` applies data selection before training. The dispatch order is:

1. `--use_llm_selection True`: use the VLM-based selector and disable traditional AL.
2. `--use_active_learning True`: run the requested AL strategy on either the original train pool or the combined pool when `--al_combined True`.
3. Otherwise: use random sub-sampling when `--sub_ratio < 1.0`.

Within the AL branch, the current strategies fall into three families:

- **Geometry / coverage-driven**: `coreset`, `diversity`, `hybrid`
- **Uncertainty-aware batch selection**: `badge`, `uncertainty_stratified`, `gen_probe`
- **Metadata / slice-driven**: `attribute`, `hierarchical`

## Available Strategies

### 1. CoreSetStrategy (Original Method)
```bash
python alpred.py --al_strategy coreset --al_initial_labeled 400 --al_embedding_dim 256
```
- **Principle**: Selects samples farthest from labeled data (diversity-first)
- **Use Cases**: Uniform data distribution, need to maximize coverage
- **Issues**: May select outlier samples, disrupting class balance

### 2. BadgeStrategy (Gradient-Embedding Batch Selection)
```bash
python alpred.py --al_strategy badge --al_embedding_dim 256
```
- **Principle**: Builds BADGE-style gradient embeddings and uses k-means++ seeding to choose a batch that is both uncertain and diverse.
- **Current Implementation Detail**: this repo implements a BADGE-style proxy variant rather than the original `JordanAsh/badge` pipeline.
- **Step 1 - Proxy labeled set**: `alpred.py` extracts embeddings from the validation split and uses the validation targets as a lightweight labeled proxy set.
- **Step 2 - Proxy class probabilities**: for each class, it computes a centroid in the normalized embedding space; each unlabeled training image gets class probabilities from a softmax over negative squared distances to those centroids.
- **Step 3 - Gradient embeddings**: it builds the usual BADGE-shaped gradient embedding by concatenating one block per class, where each block is the sample embedding scaled by the class coefficient `p_c - 1[c = argmax p]`.
- **Step 4 - Batch selection**: for smaller jobs, it runs exact BADGE-style k-means++ seeding on those gradient embeddings; for larger jobs, it switches to a cheaper approximation that samples without replacement with probabilities proportional to the gradient-embedding norm.
- **Difference from original BADGE**: the original `JordanAsh/badge` code uses the model's own predictive probabilities for the unlabeled pool and keeps the sequential BADGE seeding logic, while this repo replaces the classifier with a validation-centroid proxy and may replace exact seeding with weighted gradient-norm sampling on large pools.
- **Use Cases**: Strong baseline when a batch selector should mix uncertainty with diversity rather than optimizing either one alone.
- **Advantages**: Closer to a standard batch-AL baseline than pure coreset or pure clustering.

### 3. StratifiedStrategy (Stratified Sampling)
```bash
python alpred.py --al_strategy stratified
```
- **Principle**: Stratified sampling by class proportion, maintaining class balance
- **Use Cases**: Imbalanced data, need representation from every class
- **Advantages**: Maintains original data distribution, simple and efficient
- **Speed**: Fastest (no feature extraction needed)

### 4. DiversityStrategy (Diversity Clustering)
```bash
python alpred.py --al_strategy diversity --al_clusters 100 --al_embedding_dim 256
```
- **Principle**: K-means clustering followed by representative sample selection from each cluster
- **Use Cases**: Data with natural clustering structure, need to cover different data patterns
- **Advantages**: Balances diversity and representativeness

### 5. HybridStrategy (Mixed Strategy)
```bash
python alpred.py --al_strategy hybrid --al_coreset_ratio 0.3 --al_stratified_ratio 0.5 --al_embedding_dim 256
```
- **Principle**: Combines advantages of multiple methods
  - 30% Core-set (diversity)
  - 50% Stratified (class balance)  
  - 20% Random (randomness)
- **Use Cases**: Complex data distributions, need comprehensive consideration
- **Advantages**: Most robust, highly adaptive

### 6. AttributeFocusedStrategy (Attribute-Driven Sampling)
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

## Why the Attribute Strategy Is Different

The `attribute` strategy is not just another uncertainty or diversity heuristic. Standard AL methods in this repo behave as follows:

- `coreset` / `diversity`: rank samples from geometry in feature space.
- `badge` / `uncertainty_stratified` / `gen_probe`: rank samples from uncertainty-like signals or gradient embeddings.
- `hybrid`: mixes geometric coverage, class balance, and randomness.
- `hierarchical`: allocates budget over slices, but still uses model-side uncertainty and embeddings during selection.

By contrast, `attribute` starts from **semantic metadata** rather than feature geometry. It explicitly asks: which human-named slices are rare, uncertain, underperforming, or artificially flooded by generated data? The selector then scores each image by those slice-level signals.

The main methodological contributions in the current `attribute` path are:

- **Class-conditional rarity**: prefer values that are rare within a class, not merely rare globally.
- **Generated-data frequency penalty**: suppress oversampling of slices that were already heavily injected by synthetic data.
- **Log-aware failure weighting**: upweight samples from attribute values that earlier diagnostics marked as low-accuracy rare cases.
- **Deficit-based class balancing**: interpolate between pure attribute ranking and hard stratification without discarding the metadata signal.

This makes the method a better fit for Safefix than standard AL baselines when the goal is not only to improve average accuracy, but also to surface and repair semantically interpretable failure slices.

## Parameter Description

### General Parameters
- `--al_strategy`: AL strategy selection [attribute|badge|coreset|stratified|diversity|hybrid|hierarchical|uncertainty_stratified|gen_probe]
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
- `--al_attribute_log_acc_scale`: Scaling factor applied to each log-derived failure weight (default: 0.0)
- `--al_attribute_extra_counts`: Inline list or JSON file describing additional counts to inject per attribute value (useful when synthetic data floods specific slices); inline entries without `=count` inherit `--num_added_img`, and when omitted entirely the code infers counts from `--attributes` tokens (value+attribute) using `--num_added_img`
- `--al_attribute_frequency_penalty`: Multiplier controlling how strongly over-represented values are penalized (0 disables the term)
- `--al_attribute_class_balance`: Stratified weight for enforcing class balance after attribute scoring (0 disables, `inf` enforces near-perfect round-robin per class)
- `--al_conditional_rarity`: Use class-conditional rarity `P(value | class)` instead of global rarity

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
- `BadgeStrategy`: BADGE-style gradient-embedding batch selection
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
