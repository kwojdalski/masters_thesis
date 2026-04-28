Scan all Python files under `src/` for logger calls that violate the project logging standard. Report violations grouped by file with line numbers and the offending text. Do NOT fix anything — just report.

## Rules to enforce

1. **No f-strings** — `logger.info(f"...")` is a violation. Must use lazy `%` formatting: `logger.info("...", arg)`
2. **Lowercase start** — the message string must begin with a lowercase letter
3. **No trailing punctuation** — no `.`, `!`, or `...` at the end of the message string
4. **No ANSI codes** — no `\033[` or `\x1b[` inside message strings
5. **No prose sentences** — "Loading data from file" style is a violation; prefer `"load data path=%s"` verb-noun key=value style
6. **No shape tuples as `%s`** — `shape=%s` with a tuple arg hides the values inside a tuple repr. Unpack instead: `"n_rows=%d n_cols=%d", *df.shape`

## Search commands to run

```bash
# f-strings in logger calls
grep -rn 'logger\.\(debug\|info\|warning\|error\|critical\|exception\)(f"' src/

# Uppercase-starting messages
grep -rn 'logger\.\(debug\|info\|warning\|error\|critical\|exception\)("[A-Z]' src/

# Trailing punctuation
grep -rn 'logger\.\(debug\|info\|warning\|error\|critical\|exception\)(\"[^"]*[.!]\+"\|"\.\.\."' src/

# ANSI codes in message strings
grep -rn 'logger\.\(debug\|info\|warning\|error\|critical\|exception\)(.*\\033\[' src/

# Shape tuples passed as %s (R5)
grep -rn 'logger\.\(debug\|info\|warning\|error\|critical\|exception\)(.*shape=%s' src/
```

For each violation found, show: file path, line number, and the full logger call. Group by file. At the end, print a summary count of total violations found.
