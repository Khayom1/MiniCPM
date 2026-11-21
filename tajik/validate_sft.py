#!/usr/bin/env python3
"""Validate MiniCPM5 ShareGPT/messages JSONL for Tajik SFT."""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def cyrillic_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cyr = sum("CYRILLIC" in unicodedata.name(ch, "") for ch in letters)
    return cyr / len(letters)


def validate_file(path: Path, min_assistant_cyrillic: float) -> dict:
    stats = Counter()
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError(f"{path}:{line_no}: expected object with messages array")

            roles = []
            assistant_text = []
            conversation_signature = []
            for message in row["messages"]:
                if not isinstance(message, dict):
                    raise ValueError(f"{path}:{line_no}: each message must be an object")
                role, content = message.get("role"), message.get("content")
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(f"{path}:{line_no}: unsupported/missing role {role!r}")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"{path}:{line_no}: {role} content must be a non-empty string")
                roles.append(role)
                conversation_signature.append(f"{role}:{content.strip()}")
                if role == "assistant":
                    assistant_text.append(content)

            if "user" not in roles or "assistant" not in roles:
                raise ValueError(f"{path}:{line_no}: each example needs at least one user and assistant message")
            if roles[-1] != "assistant":
                raise ValueError(f"{path}:{line_no}: conversation must end with an assistant answer")

            signature = "\n".join(conversation_signature)
            if signature in seen:
                stats["duplicate_examples"] += 1
            seen.add(signature)

            answer = "\n".join(assistant_text)
            if cyrillic_ratio(answer) < min_assistant_cyrillic:
                stats["low_cyrillic_assistant_answers"] += 1
            stats["examples"] += 1
            stats["messages"] += len(row["messages"])

    if stats["examples"] == 0:
        raise ValueError(f"{path}: no examples found")
    return dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--min-assistant-cyrillic", type=float, default=0.55)
    args = parser.parse_args()
    if not 0 <= args.min_assistant_cyrillic <= 1:
        parser.error("--min-assistant-cyrillic must be between 0 and 1")

    try:
        report = {str(path): validate_file(path, args.min_assistant_cyrillic) for path in args.inputs}
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
