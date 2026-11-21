#!/usr/bin/env python3
"""Measure MiniCPM tokenizer fidelity and token efficiency on Tajik text."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openbmb/MiniCPM5-2B")
    parser.add_argument("--input", type=Path, help="Prepared JSONL with a text field")
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be at least 1")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Install the optional dependency first: pip install transformers", file=sys.stderr)
        return 2

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=False)
        if args.input:
            samples = []
            with args.input.open("r", encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row.get("text") if isinstance(row, dict) else None
                    if isinstance(text, str) and text.strip():
                        samples.append(text)
                    if len(samples) >= args.max_samples:
                        break
        else:
            samples = [
                "Салом! Ман мехоҳам бо забони тоҷикӣ суҳбат кунам.",
                "Илм, забон ва фарҳанги тоҷикӣ таърихи куҳан доранд.",
                "Ҳар як ҷумла бояд аз ҷиҳати маъно ва грамматика дуруст фаҳмида шавад.",
                "Ғ, ӣ, қ, ӯ, ҳ ва ҷ ҳарфҳои муҳимми алифбои тоҷикӣ мебошанд."
            ]

        if not samples:
            raise ValueError("No non-empty text samples found")

        total_chars = total_tokens = total_unk = roundtrip_exact = 0
        per_sample = []
        for sample in samples:
            ids = tokenizer.encode(sample, add_special_tokens=False)
            decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            exact = decoded == sample
            unk_id = tokenizer.unk_token_id
            unk_count = sum(token_id == unk_id for token_id in ids) if unk_id is not None else 0
            total_chars += len(sample)
            total_tokens += len(ids)
            total_unk += unk_count
            roundtrip_exact += int(exact)
            per_sample.append({
                "characters": len(sample),
                "tokens": len(ids),
                "tokens_per_character": len(ids) / max(1, len(sample)),
                "roundtrip_exact": exact,
                "unknown_tokens": unk_count
            })

        report = {
            "model": args.model,
            "tokenizer_class": tokenizer.__class__.__name__,
            "samples": len(samples),
            "total_characters": total_chars,
            "total_tokens": total_tokens,
            "tokens_per_character": total_tokens / max(1, total_chars),
            "exact_roundtrip_rate": roundtrip_exact / len(samples),
            "unknown_token_count": total_unk,
            "tajik_specific_letters_present": sorted(set("ҒғӢӣҚқӮӯҲҳҶҷ") & set("".join(samples))),
            "per_sample": per_sample,
            "note": "Unicode representability and token efficiency are not measures of semantic language understanding."
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
