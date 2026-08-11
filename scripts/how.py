"""how — project-aware LLM helper for quick CLI questions."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
MODEL = "claude-haiku-4-5-20251001"

_SYSTEM = """\
You are a fast CLI assistant for a Python machine-learning / trading-RL thesis project.
Answer concisely — one to four sentences or a short bullet list.
If the answer is a shell command, show it in a fenced code block.
Never explain what you are about to do; just answer directly.\
"""


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("usage: how <question>", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "error: ANTHROPIC_API_KEY is not set.\n"
            "Uncomment or add this line in ~/.zshrc:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("error: anthropic package missing — run: uv add --dev anthropic", file=sys.stderr)
        sys.exit(1)

    system = _SYSTEM
    if CLAUDE_MD.exists():
        system += f"\n\nProject guidelines (CLAUDE.md):\n{CLAUDE_MD.read_text(encoding='utf-8')}"

    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model=MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": question}],
    ) as stream:
        for chunk in stream.text_stream:
            print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
