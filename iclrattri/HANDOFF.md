# Handoff Notes

See `AGENTS.md` for standing editing rules.

Current context:

- This is the active ICLR 2027 migration of the AttriSelect manuscript from
  `nips26_new/`. The source NeurIPS directory remains preserved as the
  authoritative snapshot for previously reported results.
- `main.tex` now uses the bundled `iclr2027_conference` style and bibliography
  style. Keep `\iclrfinalcopy` commented for anonymous submission.
- Keep the compiled `main.pdf` in the paper root. Run
  `scripts/organize_build_artifacts.sh` after compilation to move only LaTeX
  intermediates into `build_artifacts/`.
- Do not check or enforce the 9-page limit unless the user explicitly asks;
  the user will reduce the manuscript later.
- The introduction is being shaped around generation-based repair as a promising but still inefficient direction.
- The paper's main angle is not generic repair, but efficient subset selection for generation-based repair after targeted examples have been created.
- Proposition 3 has been generalized from CoreSet/BADGE-style baselines to attribute-agnostic AL methods that do not use the current focused attributes.
- The appendix now has `Selection t-SNE Visualizations` before `Selection Overlap as an Attribute-Information Diagnostic`. It includes ratio `0.10` t-SNE figures for all four datasets, available ratio `0.05` figures for ImageNet/KITTI, and KITTI overlays for ratios `0.10` and `0.05`. Paper-local figure copies are under `figures/...`, and appendix image paths no longer point to `../figures/...`.
- The t-SNE section analyzes the expected controlled non-uniformity from the
  `R`, `P`, and class-balance terms. The last verified PDF belongs to the
  source `nips26_new/` version, not this ICLR migration.
