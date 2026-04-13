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
3. `report.md` — human-facing experiment summary and results

Do not assume chat history is available or complete. Use the Trellis workspace journal as the detailed source of truth for recent project context.

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
