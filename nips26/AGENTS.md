# Agent Instructions

This repository contains an active NeurIPS-style paper draft. The user is iteratively shaping the argument and may intentionally leave TODOs, rough wording, comments, or unfinished notes in the TeX files.

Editing rules:

- Do not edit TODOs, rough placeholders, comments, or unfinished passages unless the user explicitly asks you to modify that exact text.
- Treat provisional text as the user's working notes, not as cleanup targets.
- Keep edits narrowly scoped to the user's current request.
- Preserve nearby text unless it directly blocks the requested change.
- Do not compile unless the user asks; the user checks compilation in Overleaf.
- Prefer wording suitable for a NeurIPS submission: problem-driven motivation, careful claims, and no phrasing such as "because our experiments focus on..." as the reason for studying a problem.
- Avoid broad polishing passes unless explicitly requested.
- Keep paper-facing text at the scientific-method/result level. Do not mention internal workspace artifacts or implementation provenance such as JSON/CSV files, logs, scripts, selected-id exports, file paths, generated working directories, or private data/image sources. When needed, describe these as annotations, selected subsets, diagnostics, figures, tables, or experimental protocol. Do not add auxiliary images, datasets, or diagnostics beyond the paper's authorized main results unless the user explicitly approves them for the manuscript.
- For changes confined to the `nips26/` paper folder, do not update the Trellis journal, `.handoff.md`, or `report.md` unless the user explicitly asks for that documentation update. This saves time during rapid manuscript editing.

NeurIPS 2026 formatting memory from `Formatting Instructions.tex`:

- Main submission content is limited to 9 pages including figures; acknowledgments, references, checklist, and optional technical appendices do not count toward the content-page limit.
- Do not modify `neurips_2026.sty` or otherwise change global formatting parameters such as margins, text block size, or default paper font settings.
- At submission time, use the anonymized style without `final` or `preprint`. The `ack` environment is acceptable because it is hidden in anonymized submissions.
- Keep the abstract as one paragraph.
- Tables:
  - Center all tables.
  - Put table number/title before the table body.
  - Use `booktabs`-style rules (`\toprule`, `\midrule`, `\bottomrule`) and avoid vertical rules.
  - Keep tables neat, clean, and legible. Do not shrink tables aggressively; prefer concise columns, smaller `\tabcolsep`, or appendix placement before using very small fonts.
  - Table captions should be sentence-style capitalization except for proper nouns, and should explain what the table shows plus the key take-away when space permits.
- Figures:
  - Put figure captions after the figure.
  - Use `\includegraphics` with widths specified relative to `\linewidth` when possible.
  - Ensure figures are legible in both color and black-and-white print.
  - Avoid manual positioning commands that can break margins.
  - Figure captions should be sentence-style capitalization except for proper nouns, and should explain what the figure shows plus the key take-away when space permits.
- Math:
  - Do not use bare TeX display math (`$$ ... $$`). Use LaTeX/AMS display environments so line numbering works.
- PDF/font hygiene:
  - Prefer `pdflatex` for local PDF generation when compilation is requested.
  - Avoid packages or figures that introduce Type 3 bitmap fonts; use AMS fonts for blackboard symbols.
