# Issue tracker: GitHub

Issues and specs live in GitHub Issues for
`jdgsmallwood/cuda-spatial-filtering`. Use the `gh` CLI.

## Operations

- Before creating issues, verify the target with
  `gh repo view --json nameWithOwner`.
- Create: `gh issue create --title "..." --body-file <file>`.
- Read: `gh issue view <number> --json number,title,body,labels,comments`.
- List: `gh issue list --state open`; filter with `--label` as needed.
- Comment: `gh issue comment <number> --body-file <file>`.
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- Close: post the explanation as a comment, then `gh issue close <number>`.

Use files for multiline bodies. File issues only in this repository.

Use native GitHub sub-issues and blocking dependencies when supported.
Otherwise, put `Part of #<parent>` and `Blocked by: #<number>` at the
top of the child body and list children in the parent's task list.
A ticket is unblocked when every blocker is closed.

The installed CLI lacks `gh label`. Use `gh api` for label management.

## Skill conventions

“Publish to the issue tracker” means create a GitHub issue.
“Fetch the relevant ticket” means read the GitHub issue.

## Pull requests as a triage surface

PRs as a request surface: no.

## Wayfinding

A map is an issue labelled `wayfinder:map`. Child tickets link to the
map and use `wayfinder:research`, `wayfinder:prototype`,
`wayfinder:grilling`, or `wayfinder:task`.

Choose the first open, unassigned child in map order with no open
blockers. Claim it with `gh issue edit <number> --add-assignee @me`.
Resolve it by commenting with the answer, closing it, and adding a
summary and link to the map's Decisions-so-far section.
