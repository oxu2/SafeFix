# Handoff Notes

See `AGENTS.md` for standing editing rules.

Current context:

- The introduction is being shaped around generation-based repair as a promising but still inefficient direction.
- The paper's main angle is not generic repair, but efficient subset selection for generation-based repair after targeted examples have been created.
- Proposition 3 has been generalized from CoreSet/BADGE-style baselines to attribute-agnostic AL methods that do not use the current focused attributes.
- The appendix now has `Selection t-SNE Visualizations` before `Selection Overlap as an Attribute-Information Diagnostic`. It includes ratio `0.10` t-SNE figures for all four datasets, available ratio `0.05` figures for ImageNet/KITTI, and KITTI overlays for ratios `0.10` and `0.05`. Paper-local figure copies are under `figures/...`, and appendix image paths no longer point to `../figures/...`.
- The t-SNE section now analyzes the expected controlled non-uniformity from the `R`, `P`, and class-balance terms. `main.pdf` was recompiled with no unresolved references or citation warnings.
