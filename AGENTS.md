# SafeFix Agent Instructions

## Scope and Priority

- This file is the repo-local instruction set for this SafeFix checkout.
- Direct user instructions override this file.
- The global agent instructions apply unless this file provides a more specific rule.
- Apply repository rules only to work in this repository; an unrelated task does not activate its workflow.

---

## Context Discovery (Read What the Task Needs)

Do not preload an entire stack of project files before every edit. Read what the task needs:
- `.handoff.md`: Read when picking up current blockers, recent artifacts, or next steps.
- Active Workspace Journal: Read the active journal under `.trellis/workspace/oxu/` only when continuation details or experiment provenance are required. Discover the active journal dynamically.
- `report.md`: Read for stable, human-facing summaries of newly completed or historical local experiments; not the authority for previously reported manuscript results.
- Canonical Juno Guide: Consult `/people/cs/o/oxx220000/.codex/docs/juno_quickstart.md` for cluster rules.
- Trellis is legacy and opt-in in this workspace: do not force Trellis task initialization or channel synchronization. Small, self-contained edits do not require a project-history tour.

---

## Execution and Workflow (GPT-6 Astra Optimized)

- **Initiative and Follow-Through (Bias Towards Action)**:
  - Infer the user's intent and task scope from instructions and prior conversation context. Bias towards action and carry the user's intended task to completion.
  - When the user expresses intent to perform new work or fix an existing issue, persist until the intended goal is fulfilled. Progress autonomously towards the goal (such as editing code, creating files, fixing bugs, and running checks) unless actions are clearly destructive or irreversible.
  - Treat action requests, whether polite requests or direct action commands in English (such as "can you...", "I want to...", "help me...", "fix...", "modify...", "next...") or in Chinese (such as polite requests "帮我...", "能不能...", "可以...吗", "请把...", "我想...", as well as direct action commands "把...", "去把...", "接下来...", "进行...", "修改...", "修复...", "排查..."), as direct instructions to do the work. Do not stop at acknowledging capability or proposing an unrequested plan.
  - Complete authorized work first and prepare a concrete, reviewable result before asking for user approval.
  - Do not introduce unsolicited warnings, disclaimers, approval flows, or safety/compliance checklists due to hypothetical risk.
  - Ask questions only when a missing decision materially changes the outcome and cannot reasonably be inferred from context.
- **Proportional Verification**:
  - Calibrate testing to the scale of the change. Do not write tests for reversible, low-impact changes that mirror the implementation.
  - Run checks appropriate to the change. Once those checks pass, broaden or repeat testing only when new changes, failures, or unresolved concerns justify it; otherwise, continue toward completing the task.
- **Subagent Delegation**:
  - Delegate independent exploration, research, or validation tasks to subagents when parallel execution saves time or improves quality. Ensure messages are clear and legible.
- **Autonomous Bug Fixing**:
  - Diagnose root causes, inspect logs, and fix issues directly without asking for hand-holding.

---

## Quick Skill Routing (survives context truncation)

| Task Domain | Target Skill | Verifiable Check |
|-------------|--------------|------------------|
| Paper data charts, loss curves, grouped bars, heatmaps | `scientific-figure-making` | Publication typography, >=300 DPI, runs without warnings |
| System architecture, flowcharts, pipeline diagrams | `drawio-skill` | Vector/editable `.drawio.png` with embedded XML (`-e`) |
| Paper drafting, LaTeX submission, citation check | `ml-paper-writing` | Zero hallucinated citations; verified BibTeX records |
| Text polishing, eliminating AI boilerplate/clichés | `humanizer` | Retain exact technical terms; eliminate fluff; no contractions |

### Session Discipline & Auto-Triggers

- **New task in existing session**: Re-match against the Quick Skill Routing table. Do not depend on partial memories from earlier turns after context compaction.
- **Task Closure Verification**: Before declaring a non-trivial task complete, run the explicit verification check associated with the active skill.

---

## Current Rebuttal Editing Scope: Reviewer KaLa and Area Chair 6k27

- The user has explicitly authorized edits to both `Response to Reviewer KaLa` and `Response to Area Chair 6k27` in `draft.md`.
- KaLa clarifications may now be synchronized into the Area Chair response when they address the same concern, but preserve the scope and evidence limits of the underlying experiments.
- Partition both responses into Parts close to the 5,000-character limit. Split at a natural paragraph, table, equation, or blockquote boundary rather than forcing the boundary to align with `##` or `###` headings. No Part may exceed 5,000 characters, including its Part heading. Add another Part when needed to keep tables, equations, or blockquotes intact. Do not calculate or report KaLa or Area Chair Part character counts unless the user explicitly requests that validation.
- Across all OpenReview responses, inline backticks may be used as ordinary quotation or code-style formatting. When a direct answer, takeaway, or core conclusion uses inline backticks for emphasis, combine them with `**bold**` as the strongest emphasis. Use `**bold**` alone as the secondary emphasis level for ordinary emphasis, subsection labels, and best values in tables. Do not apply backticks or the combined strongest emphasis to ordinary table values, including runtimes. Do not place LaTeX expressions inside backticks. Use double quotation marks for exact semantic slice expressions and field assignments, such as "pink AND fabric" and "color=pink". Keep natural compound modifiers such as pink-and-fabric unquoted; do not use italics for these labels because italics belong to the emphasis hierarchy.

---

## NeurIPS 2026 Result Authority Rule

- When retrieving any result already reported in the NeurIPS 2026 manuscript, rebuttal, or appendix, use the corresponding files under `nips26_new/` as the primary and authoritative source. In particular, use `nips26_new/appendix/main_metric_tables.tex` for the official five-seed main task metrics.
- Search `nips26_new/main.tex` and the relevant `nips26_new/appendix/*.tex` section before consulting `report.md`, old logs, or workspace notes for a previously reported value, protocol, seed set, variance, or statistical claim.
- Use `report.md` to record newly completed experiments, local diagnostics, artifact paths, and historical context. New results may be added there, but older `report.md` values must not overwrite, invalidate, or silently replace an existing value in `nips26_new/`.
- If a new experiment is compared with a manuscript result, label differences in seeds, aggregation, code state, generator, checkpoint, or protocol. Do not present a single-seed local result and a five-seed manuscript mean as a paired or statistically matched comparison.
- If local artifacts and `nips26_new/` disagree, preserve both records, treat the TeX manuscript as authoritative, and document the discrepancy in the workspace journal and `.handoff.md`.

---

## Repo-Specific Documentation Update Rule

After any meaningful implementation, modification, or completed investigation that changes the repo state or future-agent understanding:

1. Update the active workspace journal under `.trellis/workspace/oxu/` with what changed, why it changed, key files, commands, outputs, or artifacts, and important caveats and next steps.
2. Update `.handoff.md` only when the active baton-pass state changed, such as current goal, blocker, latest useful artifact or result, and exact next step or command.
3. Update `report.md` only for stable, human-facing results worth preserving.
4. Do not leave meaningful code, workflow, experiment, or configuration changes undocumented if they matter for continuation by another agent.
5. If no documentation file needs updating, be prepared to state why.
6. Draw.io-only edits do not need to be recorded in the workspace journal. This includes changes to `.drawio`, `.drawio.xml`, and generated Draw.io exports such as `.png`, `.svg`, or `.pdf`, unless the user explicitly asks for journal documentation or the edit changes non-figure project context.
7. **2/3 Recording Threshold for Lessons and Rules**: When considering adding a new rule or lesson to `tasks/lessons.md` or a skill's `gotchas.md`, apply the After-Action Review scan (New pattern? New trap? Missing rule? Outdated rule?). Record only if at least 2 of the following 3 criteria are satisfied:
   - *Repeatable*: Likely to be encountered in future tasks by different agents or models.
   - *High Cost*: Causes significant debugging or revision time (>20 minutes) if unknown.
   - *Invisible in Code*: Cannot be inferred directly from existing code or standard framework manuals.

---

## Workspace Symlink Fallback Rule

- Treat each workspace symlink to a regular file as the primary shortcut and preserve the symlink path.
- Keep a same-directory fallback snapshot beside it by inserting `_local` before the filename extension: `file.md` -> `file_local.md`, `file.py` -> `file_local.py`, and extensionless `file` -> `file_local`.
- If a symlink already points to the corresponding `_local` file, that existing file is the fallback and no additional `_local_local` copy is needed.
- When the canonical target is changed, refresh the matching `_local` snapshot in the same change when the fallback should remain current. Use the symlink first; use the local snapshot only when the target is unavailable.
- Never replace or remove the primary symlink merely to edit its contents. Check all workspace symlinks, including links under `docs/` and `scripts/`, before adding or changing fallback files.

---

## Repo-Specific GPU Assignment Rule

- Default new Juno GPU experiment submissions to the `h200` partition. Use `h100` only as the secondary fallback or when the user explicitly requests it. Do not default to A30 or virtual-A30 partitions.
- For future independent experiment shards with the same code, environment, model, and resource shape, use one SLURM array. For this user's Juno GPU workloads, default to `--qos=juno-pri` and `%8` concurrency, for example `#SBATCH --array=0-N%8` or `sbatch --qos=juno-pri --array=0-N%8 ...`. Recheck the live association and QOS limits before submission; use normal QOS and `%4` only as a fallback or explicit conservative control. Do not default these workloads to one-GPU sequential execution unless task dependencies or an explicit user instruction require it.
- Read `/people/cs/o/oxx220000/.codex/docs/juno_quickstart.md` as the only Juno workflow source. It contains both shared rules and the SafeFix project profile.
- Repo-local `docs/juno_quickstart.md` is a convenience symlink to the canonical Codex file. It may be opened or edited from the repo, but do not replace it with an independent copy.

When the user explicitly tells the agent to run a job on a specific GPU, do not spend time checking whether that GPU is available first. Assume the user has already verified that the named GPU is ready, and launch directly on the requested GPU unless the user asks for a check.

---

## Repo-Specific Figure-Making Rule

When creating or revising publication-style figures in this repo, use the installed Codex skill `scientific-figure-making` by default. This applies to matplotlib charts, heatmaps, grouped bars, trend plots, multi-panel figures, and paper/report figure exports. For figures inside `nips26_new/`, also follow `nips26_new/AGENTS.md`.
