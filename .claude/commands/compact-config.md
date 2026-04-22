Review and clean up the project's Claude configuration after a bootstrapping session.

Steps to perform:
1. Read all files in `.claude/commands/` and the current `CLAUDE.md`.
2. Identify commands that:
   - Duplicate each other or overlap significantly
   - Are too thin (just a single obvious command with no real guidance)
   - Were added speculatively but haven't been used
3. For each redundant or weak command, propose either removing it or merging it into another.
4. Review `CLAUDE.md` for sections that are now stale, overly verbose, or covered by a skill — propose trimming.
5. Check `.claude/settings.json` for unused hooks or permissions.
6. Present a concrete diff/plan to the user before making any changes.

Goal: leave the config lean — every command should earn its place.
