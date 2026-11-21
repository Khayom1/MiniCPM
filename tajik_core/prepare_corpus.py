#!/usr/bin/env python3
"""Build a deduplicated Tajik Cyrillic corpus from TXT, JSON, and JSONL sources."""
from __future__ import annotations
import argparse, hashlib, json, random, re, unicodedata
from pathlib import Path
from typing import Any, Iterable

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
SPACE_RE = re.compile(r"[ \t]+")
PARAGRAPH_RE = re.compile(r"\n{3,}")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\ufeff", "").replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_RE.sub(" ", text)
    text = PARAGRAPH_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()

def is_usable_tajik(text: str, min_chars: int) -> bool:
    if len(text) < min_chars:
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(bool(CYRILLIC_RE.fullmatch(c)) for c in letters) / len(letters) >= 0.55

def extract_text(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        if isinstance(value.get("text"), str):
            yield value["text"]
        messages = value.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") in {"user", "assistant", "system"}:
                    content = message.get("content")
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                yield part["text"]
        for key in ("instruction", "input", "output", "question", "answer"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                yield item
    elif isinstance(value, list):
        for item in value:
            yield from extract_text(item)

def read_file(path: Path) -> Iterable[str]:
    if path.suffix.lower() == ".txt":
        yield path.read_text(encoding="utf-8", errors="replace")
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield from extract_text(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    elif path.suffix.lower() == ".json":
        yield from extract_text(json.loads(path.read_text(encoding="utf-8")))

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare TAJIK-CORE corpus")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--validation-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    if not a.input.exists(): p.error(f"Input does not exist: {a.input}")
    if a.min_chars < 1 or not 0 <= a.validation_ratio < 0.5: p.error("Invalid min-chars or validation-ratio")
    sources = [a.input] if a.input.is_file() else sorted(
        x for x in a.input.rglob("*") if x.is_file() and x.suffix.lower() in {".txt", ".json", ".jsonl"}
    )
    seen, documents = set(), []
    stats = {"files": len(sources), "records_seen": 0, "accepted": 0, "duplicates": 0, "rejected": 0}
    for source in sources:
        for raw in read_file(source):
            stats["records_seen"] += 1
            text = normalize(raw)
            if not is_usable_tajik(text, a.min_chars):
                stats["rejected"] += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                stats["duplicates"] += 1
                continue
            seen.add(digest)
            documents.append(text)
            stats["accepted"] += 1
    if not documents: raise SystemExit("No usable Tajik documents found; no corpus written.")
    random.Random(a.seed).shuffle(documents)
    n_val = max(1, round(len(documents) * a.validation_ratio)) if len(documents) > 1 and a.validation_ratio else 0
    a.output_dir.mkdir(parents=True, exist_ok=True)
    def write(path: Path, rows: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as out:
            for row in rows: out.write(json.dumps({"text": row}, ensure_ascii=False) + "\n")
    write(a.output_dir / "train.jsonl", documents[n_val:])
    write(a.output_dir / "validation.jsonl", documents[:n_val])
    manifest = {"schema_version": 1, "language": "tg", "script": "Cyrillic", "seed": a.seed,
                "validation_ratio": a.validation_ratio, "training_documents": len(documents)-n_val,
                "validation_documents": n_val, "statistics": stats}
    (a.output_dir / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
