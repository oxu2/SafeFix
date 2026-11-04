# AttriSelect: CVPR 2027 draft

This directory contains a CVPR 2027 review-format migration of the complete
`27iclr/` manuscript, including its bibliography, figures, and appendices.
The manuscript text and existing experimental claims are carried over; the
layout has been adapted to the two-column template. The bundled `cvpr.sty`
and `ieeenat_fullname.bst` are copied from `templete cvpr2027/`.

Compile from this directory with a TeX installation containing `latexmk`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error AttriSelect.tex
bash scripts/organize_build_artifacts.sh
```

`AttriSelect.pdf` combines the main paper, references, and supplementary material.
The supplementary material starts on its own page; separate it from the main
paper when preparing submission files. This migration preserves the complete
source content without shortening it for any page limit. Some inherited figure
PDFs contain Type 3 fonts; check figure and submission requirements before
final submission.
