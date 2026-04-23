Capture a workflow as a new project slash command.

When invoked, ask the user:
1. What workflow or repeated action should this skill automate?
2. What should the slash command be named (short, hyphenated, verb-first)?
3. Any arguments or variants it should handle?

Then create `.claude/commands/<name>.md` with:
- A one-line description of what it does
- The exact command(s) to run, with `$ARGUMENTS` placeholder if applicable
- Usage examples

After writing the file, note it as a candidate for review during `/compact-config`.
