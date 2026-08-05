<!-- TRELLIS:START -->
# Trellis Instructions

These instructions are for AI assistants working in this project.

Use the `/trellis:start` command when starting a new session to:
- Initialize your developer identity
- Understand current project context
- Read relevant guidelines

Use `@/.trellis/` to learn:
- Development workflow (`workflow.md`)
- Project structure guidelines (`spec/`)
- Developer workspace (`workspace/`)

Keep this managed block so 'trellis update' can refresh the instructions.

<!-- TRELLIS:END -->

## Repo-Specific Agent Context Note

Before continuing work in a new session, read these files in order:

1. `.trellis/workspace/oxu/journal-1.md` — detailed recent multi-agent context, decisions, outputs, and caveats
2. `.handoff.md` — short cross-session / cross-account handoff summary
3. `report.md` — human-facing summaries of newly completed or historical local experiments; not the authority for previously reported manuscript results

Do not assume chat history is available or complete. Use the Trellis workspace journal as the detailed source of truth for recent project context.

## Background Agent Delegation

- The user authorizes the primary agent to decide when to use background
  agents without requesting separate permission each time.
- Proactively use background agents for independent work that benefits from
  parallel execution, such as research, code search, and validation.
- Handle simple tasks directly and avoid delegation when it would add
  unnecessary coordination overhead.

## Current Rebuttal Editing Scope: Reviewer KaLa and Area Chair 6k27

- The user has explicitly authorized edits to both `Response to Reviewer KaLa`
  and `Response to Area Chair 6k27` in `draft.md`.
- KaLa clarifications may now be synchronized into the Area Chair response when
  they address the same concern, but preserve the scope and evidence limits of
  the underlying experiments.
- Partition both responses into Parts close to the 5,000-character limit.
  Split at a natural paragraph, table, equation, or blockquote boundary rather
  than forcing the boundary to align with `##` or `###` headings. No Part may
  exceed 5,000 characters, including its Part heading. Add another Part when
  needed to keep tables, equations, or blockquotes intact. Do not calculate or
  report KaLa or Area Chair Part character counts unless the user explicitly
  requests that validation.
- Across all OpenReview responses, inline backticks may be used as ordinary
  quotation or code-style formatting. When a direct answer, takeaway, or core
  conclusion uses inline backticks for emphasis, combine them with `**bold**`
  as the strongest emphasis. Use `**bold**` alone as the secondary emphasis
  level for ordinary emphasis, subsection labels, and best values in tables.
  Do not apply backticks or the combined strongest emphasis to ordinary table
  values, including runtimes. Do not place LaTeX expressions inside backticks.
  Use double quotation marks for exact semantic slice expressions and field
  assignments, such as "pink AND fabric" and "color=pink". Keep natural
  compound modifiers such as pink-and-fabric unquoted; do not use italics for
  these labels because italics belong to the emphasis hierarchy.

## NeurIPS 2026 Result Authority Rule

- When retrieving any result already reported in the NeurIPS 2026 manuscript,
  rebuttal, or appendix, use the corresponding files under `nips26_new/` as
  the primary and authoritative source. In particular, use
  `nips26_new/appendix/main_metric_tables.tex` for the official five-seed main
  task metrics.
- Search `nips26_new/main.tex` and the relevant
  `nips26_new/appendix/*.tex` section before consulting `report.md`, old logs,
  or workspace notes for a previously reported value, protocol, seed set,
  variance, or statistical claim.
- Use `report.md` to record newly completed experiments, local diagnostics,
  artifact paths, and historical context. New results may be added there, but
  older `report.md` values must not overwrite, invalidate, or silently replace
  an existing value in `nips26_new/`.
- If a new experiment is compared with a manuscript result, label differences
  in seeds, aggregation, code state, generator, checkpoint, or protocol. Do not
  present a single-seed local result and a five-seed manuscript mean as a
  paired or statistically matched comparison.
- If local artifacts and `nips26_new/` disagree, preserve both records, treat
  the TeX manuscript as authoritative, and document the discrepancy in the
  Trellis journal and `.handoff.md`.

## Repo-Specific Documentation Update Rule

After any meaningful implementation, modification, or completed investigation that changes the repo state or future-agent understanding:

1. Update `.trellis/workspace/oxu/journal-1.md` with:
   - what changed
   - why it changed
   - key files, commands, outputs, or artifacts
   - important caveats and next steps
2. Update `.handoff.md` only when the active baton-pass state changed, such as:
   - current goal
   - blocker
   - latest useful artifact or result
   - exact next step or command
3. Update `report.md` only for stable, human-facing results worth preserving.
4. Do not leave meaningful code, workflow, experiment, or configuration changes undocumented if they matter for continuation by another agent.
5. If no documentation file needs updating, be prepared to state why.
6. Draw.io-only edits do not need to be recorded in `.trellis/workspace/oxu/journal-1.md`. This includes changes to `.drawio`, `.drawio.xml`, and generated Draw.io exports such as `.png`, `.svg`, or `.pdf`, unless the user explicitly asks for journal documentation or the edit changes non-figure project context.

## Repo-Specific GPU Assignment Rule

- Default new Juno GPU experiment submissions to the `h200` partition. Use
  `h100` only as the secondary fallback or when the user explicitly requests
  it. Do not default to A30 or virtual-A30 partitions.
- For future independent experiment shards with the same code, environment,
  model, and resource shape, use one SLURM array. Through August 10, 2026, add
  `--qos=juno-pri` and use `%8` concurrency whenever the array contains more
  than eight tasks, for example `#SBATCH --array=0-N%8` or
  `sbatch --qos=juno-pri --array=0-N%8 ...`. Do not default these workloads to
  one-GPU sequential execution unless task dependencies or an explicit user
  instruction require it. After the temporary QoS is removed, return to `%4`
  concurrency unless newer guidance replaces this rule.
- Read `/people/cs/o/oxx220000/.codex/docs/juno_quickstart.md` as the only
  Juno workflow source. It contains both shared rules and the SafeFix project
  profile.
- Repo-local `docs/juno_quickstart.md` is a convenience symlink to the
  canonical Codex file. It may be opened or edited from the repo, but do not
  replace it with an independent copy.

When the user explicitly tells the agent to run a job on a specific GPU, do not spend time checking whether that GPU is available first. Assume the user has already verified that the named GPU is ready, and launch directly on the requested GPU unless the user asks for a check.

## Repo-Specific Figure-Making Rule

When creating or revising publication-style figures in this repo, use the installed Codex skill `scientific-figure-making` by default. This applies to matplotlib charts, heatmaps, grouped bars, trend plots, multi-panel figures, and paper/report figure exports. For figures inside `nips26/`, also follow `nips26/AGENTS.md`.
