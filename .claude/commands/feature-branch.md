# Feature Branch Setup

Start feature work or prepare a branch for a PR, always keeping personal Claude config commits out of upstream.

## Usage

```
/feature-branch <description of work>
/feature-branch --pr   # prepare current branch for PR (rebase + verify)
```

## Steps

### Starting new feature work

Given `$ARGUMENTS`:

1. Derive a kebab-case branch name from the description (e.g., "add retry logic" → `feature/add-retry-logic`)
2. Create a new git worktree branched from `claude/config`:
   ```bash
   git worktree add -b <branch-name> .claude/worktrees/<branch-name> claude/config
   ```
3. Report the worktree path so the user can navigate to it or tell you to enter it.

If already inside a worktree and just need a new branch (no new worktree), do:
```bash
git checkout -b <branch-name> claude/config
```

### Preparing for PR (`--pr` flag or when user says they're ready to open a PR)

1. Check for config commits on the current branch:
   ```bash
   git log main..claude/config --format="%H"
   # then check each against: git merge-base --is-ancestor <hash> HEAD
   ```
2. If any config commits found, rebase onto main:
   ```bash
   git rebase main
   ```
3. Verify no config commits remain:
   ```bash
   git log main..HEAD --oneline
   ```
   Confirm the output shows only feature commits, not CLAUDE.md/.claude/ changes from the config branch.
4. Push:
   ```bash
   git push -u origin <branch-name>
   ```
5. Optionally open the PR with `gh pr create`.

## Notes

- Always branch from `claude/config`, never from `main` directly — this ensures personal Claude settings (CLAUDE.md, .claude/) are present while working.
- The PreToolUse hook on `git push` will block if you forget the rebase — but run it proactively anyway.
- After rebase, the local branch still has your work; only the config commits are stripped from the upstream-facing history.
