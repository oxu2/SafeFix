# Agent Instructions

This directory contains the active ICLR 2027 AttriSelect paper draft migrated from `nips26_new`. The user is iteratively shaping the argument and may intentionally leave TODOs, rough wording, comments, or unfinished notes in the TeX files.

Editing rules:

- When editing `.tex` files in this folder, use the installed Codex skills
  `ml-paper-writing` and `humanizer` by default. Apply `ml-paper-writing` for
  ICLR-style argument structure, claims, citations, captions, and LaTeX
  paper conventions; apply `humanizer` as a final pass to remove formulaic
  AI-generated wording while preserving the paper's technical meaning and
  formal academic tone.
- When creating or revising paper figures from this folder, also use the
  installed `academic-plotting` skill, and prefer the existing
  `scientific-figure-making` style when producing matplotlib figures.
- Do not edit TODOs, rough placeholders, comments, or unfinished passages unless the user explicitly asks you to modify that exact text.
- Treat provisional text as the user's working notes, not as cleanup targets.
- Keep edits narrowly scoped to the user's current request.
- Preserve nearby text unless it directly blocks the requested change.
- Do not compile unless the user asks; the user checks compilation in Overleaf.
- Do not run page-count checks or shorten the manuscript to meet the ICLR
  page limit unless the user explicitly asks. The user will handle later
  page-budget reduction.
- Prefer wording suitable for an ICLR submission: problem-driven motivation, careful claims, and no phrasing such as "because our experiments focus on..." as the reason for studying a problem.
- Avoid broad polishing passes unless explicitly requested.
- Keep paper-facing text at the scientific-method/result level. Do not mention internal workspace artifacts or implementation provenance such as JSON/CSV files, logs, scripts, selected-id exports, file paths, generated working directories, or private data/image sources. When needed, describe these as annotations, selected subsets, diagnostics, figures, tables, or experimental protocol. Do not add auxiliary images, datasets, or diagnostics beyond the paper's authorized main results unless the user explicitly approves them for the manuscript.
- For changes confined to the `nips26/` paper folder, do not update the Trellis journal, `.handoff.md`, or `report.md` unless the user explicitly asks for that documentation update. This saves time during rapid manuscript editing.

ICLR 2027 formatting rules from the bundled template:

- The initial submission has a strict 9-page main-text limit, with unlimited additional pages for references. The rebuttal/camera-ready limit is 10 pages.
- Do not modify `iclr2027_conference.sty` or otherwise change global formatting parameters such as margins, text block size, or default paper font settings.
- Keep `\iclrfinalcopy` commented for anonymous submission. Uncomment it only for camera-ready.
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

## Specific Paper Formatting and Styling Decisions

To maintain a consistent, premium academic design throughout this paper (and future projects), adhere to the following decisions:

1. **Model Name (`\modelname`, `\model`, `\modelplain`)**:
   - **Single Point of Control**: Define three macros to make name changes easy while controlling color:
     ```latex
     \newcommand{\modelname}{AttriSelect} % Raw text string
     \newcommand{\model}{\textcolor{darkteal}{\textbf{\modelname}}\xspace} % Styled name for body text
     \newcommand{\modelplain}{\modelname\xspace} % Plain text name for headings
     ```
   - **Body Text**: Use `\model` in body text to apply the custom dark teal color and bolding (`\definecolor{darkteal}{HTML}{005F73}`).
   - **Structural Headings**: Do **NOT** use the colored `\model` macro in structural headings (e.g., `\title`, `\section`, `\subsection`, `\subsubsection`). Instead, use the plain-text `\modelplain` macro to keep headers and the Table of Contents completely black, while allowing the model name to be changed in a single place.

2. **Hyperlinks (`hyperref`)**:
   - **Single-Color Harmony**: Maintain unified hyperlink colors (`linkcolor=myblue`, `citecolor=myblue`, `urlcolor=myblue` with `\definecolor{myblue}{HTML}{268BD2}`). This provides a modern, clean aesthetic and prevents multicolored clutter.
   - **Clickability Distinction**: Ensure the model name color (`darkteal`) is noticeably darker than the hyperlink color (`myblue`) so readers do not mistake the model name for a clickable hyperlink.

3. **Table Formatting**:
   - **Column Highlighting**: Highlight the proposed method's column by specifying `>{\columncolor{myblue!8}}` in the tabular layout (e.g., `\begin{tabular}{l c ... >{\columncolor{myblue!8}}c}`). This uses a light tint of the custom theme color `myblue` to keep a unified aesthetic instead of generic default `blue!5`.
   - **Cell Highlighting**: Do **NOT** use multicolored/pink backgrounds for individual best-performing cells. Keep best results clean and simply **bold** them (`\textbf{}`).
   - **Sub-group Separations (Dashed Lines)**: Use dense dashed lines to separate sub-groups (such as ratios `0.9` and `1.0`). Use `\usepackage{arydshln}` and tighten them globally in the preamble:
     ```latex
     \setlength{\dashlinedash}{1.5pt}
     \setlength{\dashlinegap}{1.5pt}
     ```
     Use `\cdashline{3-10}` at division points, skipping `\multirow` and empty columns.
     *Note*: If `\columncolor` (from `colortbl`) overlaps and hides the dashed line in the colored column, wrap the `\cdashline` inside a vertical spacing block to prevent layering overlap:
     ```latex
     \noalign{\vskip 1.5pt}\cdashline{3-10}\noalign{\vskip 1.5pt}
     ```

4. **Theoretical Elements & Proofs**:
   - **Boxes**: Keep Proposition and Theorem boxes clean (do not wrap Proposition body in boxes).
   - **Appendix/Proof Redirection**: Emphasize directions to the appendix by wrapping the redirection sentence at the end of the section in a very light grey, low-padding card:
     ```latex
     \begin{tcolorbox}[
         colback=gray!3!white, 
         colframe=gray!15!white, 
         arc=2pt, 
         boxrule=0.5pt, 
         top=4pt, bottom=4pt, left=6pt, right=6pt
     ]
     \small\noindent See Appendix~\ref{app:theory-analysis} for the complete proof...
     \end{tcolorbox}
     ```

5. **Algorithms & Cards**:
   - **Algorithm Layout**: Present pseudocode inside a floating figure wrapped in a custom card. The body of the algorithm uses a clean white background, while the title header block uses a distinct background color (e.g., `myblue` or its related colors) to stand out:
     ```latex
     \begin{tcolorbox}[
         colback=white,
         colframe=myblue!75!black,
         colbacktitle=myblue!12!white,
         coltitle=myblue!60!black,
         title=\textbf{Algorithm 1: ...},
         fonttitle=\bfseries,
         rounded corners,
         drop shadow
     ]
     ...
     \end{tcolorbox}
     ```

6. **Glossary & Notations**:
   - **Glossary Placement**: Place the Glossary of variables and symbols as the first section in the Appendix (e.g., `\section{Glossary}`).
   - **Glossary Table**: Summarize the notations in a dedicated table labeled `tab:glossary` with the caption: `Glossary of variables and symbols used in this paper.` using standard LaTeX booktabs guidelines.

---

## Directory Layout Convention

```
iclrattri/
├── main.tex              # Main manuscript
├── math_commands.tex     # Math macro definitions
├── iclr2027_conference.sty # ICLR style file (do not modify)
├── iclr2027_conference.bst # ICLR bibliography style
├── ref.bib               # Bibliography
├── figures/              # All figures/PDFs used in the paper
├── scripts/              # One-off Python fix/migration scripts
├── build_artifacts/      # LaTeX build outputs (aux, log, bbl, fls, …)
└── test_files/           # Scratch .tex test files and their build outputs
```

---

## LaTeX Table Formatting Rules (Session 2026-07-14)

These rules were applied during a formatting pass on `main.tex` and should be
reused when compatible with the bundled ICLR template.

### 7. Table Vertical Spacing — Colored Header Row Fills Entire Height

When a `\rowcolor` is applied to the header row and the table uses `booktabs`
rules, the default `booktabs` rule padding (above/below) creates a white gap
that cuts through the colored background, making the header look incomplete.

**Fix**: set the rule separation to zero globally (in the preamble, right after
`\usepackage{booktabs}`), then compensate with a modest `\arraystretch`:

```latex
\usepackage{booktabs}
\setlength{\aboverulesep}{0pt}
\setlength{\belowrulesep}{0pt}
\renewcommand{\arraystretch}{1.2}
\usepackage{colortbl}
```

- `\aboverulesep` / `\belowrulesep` control the vertical padding that `booktabs`
  inserts around `\toprule`, `\midrule`, `\bottomrule`. Setting them to `0pt`
  removes the white gaps.
- `\arraystretch` is raised slightly to maintain comfortable row height now that
  the rule padding is gone.
- If a specific table needs tighter or looser spacing, override locally inside
  the `table` environment with another `\renewcommand{\arraystretch}{...}`.

### 8. "Ours" Column Header Cell Color — Use Column Background, Not Gray

When a table highlights the proposed-method column with
`>{\columncolor{myblue!8}}c`, the header row cell for that column must **also**
use `\cellcolor{myblue!8}` (not `\cellcolor{sectiongray}`).

Reason: using `sectiongray` breaks the visual continuity of the blue column and
makes the header look detached from the data rows below.

**Pattern** (header row of every table that has a blue Ours column):

```latex
\rowcolor{sectiongray}
  Col1 & Col2 & \cellcolor{myblue!8}\textbf{Ours} \\
```

Apply the same rule when there are **two adjacent Ours columns** (e.g.,
`Ours (ModelA)` and `Ours (ModelB)`): give each its own `\cellcolor{myblue!8}`.

### 9. Separating Adjacent "Ours" Columns with a Vertical Rule

When a table contains two adjacent blue columns representing two variants of
the proposed method (e.g., `Ours (InstructPix2Pix)` vs. `Ours (ControlNet)`,
or `Ours (ICEdit)` vs. `Ours (ControlNet)`), add a vertical separator line
between them so readers can distinguish the two columns at a glance.

**How**: insert a bare `|` between the two column specifiers in the `\begin{tabular}` declaration:

```latex
\begin{tabular}{c c c c c c c c
    >{\columncolor{myblue!8}}c | >{\columncolor{myblue!8}}c}
```

The `|` is placed between the two colored column specifiers. This is the one
case where a vertical rule is acceptable in an otherwise `booktabs`-style table,
because it communicates a structural distinction (different generator/model)
rather than decorative separation.

### 10. Colored Math Symbols for Multi-Term Score Components

When a loss or score has named components ($R$, $P$, $C$, etc.), assign each a
distinct, semantically motivated color and apply it **consistently** in
equations, inline text, algorithm pseudocode, and ablation tables:

```latex
% Preamble color definitions
\definecolor{colorR}{HTML}{D79B01}   % Rarity (warm amber)
\definecolor{colorP}{HTML}{D62728}   % Frequency Penalty (red)
\definecolor{colorC}{HTML}{9573A6}   % Class Balance (purple)
```

Usage in text / equations:

```latex
$\textcolor{colorR}{R_{i,a}}$   % rarity term
$\textcolor{colorP}{P_{i,a}}$   % frequency penalty term
$\textcolor{colorC}{C_c^{(k)}}$ % class balance term
```

In ablation table headers, use `\boldsymbol` for bold math in the "full model"
column, wrapped in the same `\textcolor`:

```latex
\cellcolor{myblue!8}$\textcolor{colorR}{\boldsymbol{R}}
    +\textcolor{colorP}{\boldsymbol{P}}
    +\textcolor{colorC}{\boldsymbol{C}}$
```

Color individual cells and header labels, **not** entire rows, so only the
relevant symbols are colored.

### 11. Section Header Rows in Tables — `sectiongray` Rowcolor

Use `\rowcolor{sectiongray}` on the header row of every table to produce a
subtle light-gray band that anchors the column labels visually:

```latex
\definecolor{sectiongray}{gray}{0.90}
```

Then in the table:

```latex
\toprule
\rowcolor{sectiongray}
  \textbf{Dataset} & \textbf{Col1} & \ldots & \cellcolor{myblue!8}\textbf{Ours} \\
\midrule
```

Do **not** apply `\rowcolor{sectiongray}` to data rows. It is reserved for
column-label rows only, keeping the table uncluttered.

### 12. Dense Dashed Sub-group Separators

For tables that contain a "ratio = 1.0" row representing a special boundary
(e.g., full-data baseline vs. limited-data rows), use a dense dashed `\cdashline`
to visually separate the two groups *without* taking up a full line:

```latex
% Preamble
\usepackage{arydshln}
\setlength{\dashlinedash}{1.5pt}
\setlength{\dashlinegap}{1.5pt}
```

At the division point in the table body:

```latex
\noalign{\vskip 1.5pt}\cdashline{3-10}\noalign{\vskip 1.5pt}
```

- Skip columns occupied by `\multirow` cells (typically columns 1–2) to avoid
  breaking through the spanning text.
- Wrap with `\noalign{\vskip 1.5pt}` to avoid overlap with `\columncolor`
  backgrounds that can otherwise hide the dashed line.

---

## Proof / Appendix Redirect Box

```latex
\begin{tcolorbox}[
    colback=gray!3!white,
    colframe=gray!15!white,
    arc=2pt,
    boxrule=0.5pt,
    top=4pt, bottom=4pt, left=6pt, right=6pt
]
\small\noindent \textbf{Proof.} See Appendix~\ref{app:...} for the complete proof.
\end{tcolorbox}
```

The word **Proof** in the redirect sentence should be **bold** so readers can
find it quickly when scanning.
