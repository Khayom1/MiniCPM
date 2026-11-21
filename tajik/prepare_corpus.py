#!/usr/bin/env python3
"""Prepare auditable Tajik text JSONL into deterministic train/validation/test splits.

This performs Unicode and script checks, not full Tajik language identification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DEFAULT_TEXT_FIELDS = ("tajik", "text", "content", "body", "document")
DEFAULT_LANG_FIELDS = ("lang", "language", "locale")
TAJIK_LANGUAGE_TAGS = {"tg", "tg-tj", "tajik", "tajiki", "тоҷикӣ"}
TAJIK_SPECIFIC_LETTERS = set("ҒғӢӣҚқӮӯҲҳҶҷ")


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace without collapsing distinct Tajik letters."""
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    value = "".join(ch for ch in value if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def script_stats(text: str) -> dict[str, float | int]:
    letters = [ch for ch in text if ch.isalpha()]
    total = len(letters)
    cyrillic = sum("CYRILLIC" in unicodedata.name(ch, "") for ch in letters)
    arabic = sum("ARABIC" in unicodedata.name(ch, "") for ch in letters)
    latin = sum("LATIN" in unicodedata.name(ch, "") for ch in letters)
    tajik_markers = sum(ch in TAJIK_SPECIFIC_LETTERS for ch in letters)
    denominator = total or 1
    return {
        "letters": total,
        "cyrillic": cyrillic,
        "arabic": arabic,
        "latin": latin,
        "tajik_marker_letters": tajik_markers,
        "cyrillic_ratio": cyrillic / denominator,
        "arabic_ratio": arabic / denominator,
        "latin_ratio": latin / denominator,
    }


def get_first_string(row: dict[str, Any], fields: Iterable[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def language_is_tajik(row: dict[str, Any], lang_fields: tuple[str, ...], assume_language: str | None) -> bool:
    raw = get_first_string(row, lang_fields)
    if raw is None:
        return (assume_language or "").strip().lower() in TAJIK_LANGUAGE_TAGS
    normalized = raw.strip().lower().replace("_", "-")
    return normalized in TAJIK_LANGUAGE_TAGS


def stable_split(doc_id: str, seed: str, validation_percent: int, test_percent: int) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + validation_percent:
        return "validation"
    return "train"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(
    inputs: list[Path],
    output_dir: Path,
    text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS,
    lang_fields: tuple[str, ...] = DEFAULT_LANG_FIELDS,
    assume_language: str | None = None,
    min_alpha_chars: int = 80,
    min_cyrillic_ratio: float = 0.55,
    max_arabic_ratio: float = 0.12,
    max_latin_ratio: float = 0.35,
    validation_percent: int = 5,
    test_percent: int = 5,
    seed: str = "tajik-v1",
    overwrite: bool = False,
    lid_model_path: Path | None = None,
    lid_threshold: float = 0.70,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("At least one --inputs file is required")
    if validation_percent < 0 or test_percent < 0 or validation_percent + test_percent >= 100:
        raise ValueError("validation/test percentages must be nonnegative and sum to less than 100")
    if not 0 <= min_cyrillic_ratio <= 1 or not 0 <= max_arabic_ratio <= 1 or not 0 <= max_latin_ratio <= 1:
        raise ValueError("script ratio thresholds must be between 0 and 1")
    if min_alpha_chars < 1 or not 0 <= lid_threshold <= 1:
        raise ValueError("min_alpha_chars must be positive and lid_threshold must be between 0 and 1")

    lid_model = None
    if lid_model_path is not None:
        try:
            import fasttext
        except ImportError as exc:
            raise ValueError("Install optional dependency with: pip install fasttext-wheel") from exc
        lid_model = fasttext.load_model(str(lid_model_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{name}.jsonl" for name in ("train", "validation", "test")]
    manifest_path = output_dir / "manifest.json"
    if not overwrite and any(path.exists() for path in [*outputs, manifest_path]):
        raise FileExistsError("Output exists; pass --overwrite to replace prepared files")

    seen: set[str] = set()
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    rejected: Counter[str] = Counter()
    aggregates: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    lid_confidence_sum = 0.0
    lid_checked = 0
    for input_path in inputs:
        for row in iter_jsonl(input_path):
            aggregates["records_seen"] += 1
            if not language_is_tajik(row, lang_fields, assume_language):
                rejected["language_missing_or_not_tajik"] += 1
                continue
            raw = get_first_string(row, text_fields)
            if raw is None:
                rejected["text_field_missing"] += 1
                continue
            text = normalize_text(raw)
            if not text:
                rejected["empty_after_normalization"] += 1
                continue
            if len(text) < min_alpha_chars:
                rejected["too_short"] += 1
                continue

            stats = script_stats(text)
            if stats["cyrillic_ratio"] < min_cyrillic_ratio:
                rejected["low_cyrillic_ratio"] += 1
                continue
            if stats["arabic_ratio"] > max_arabic_ratio:
                rejected["high_arabic_ratio"] += 1
                continue
            if stats["latin_ratio"] > max_latin_ratio:
                rejected["high_latin_ratio"] += 1
                continue
            if lid_model is not None:
                labels, probabilities = lid_model.predict(text.replace("\n", " ")[:10000], k=2)
                predicted = labels[0].replace("__label__", "") if labels else ""
                confidence = float(probabilities[0]) if len(probabilities) else 0.0
                lid_checked += 1
                lid_confidence_sum += confidence
                if predicted != "tg" or confidence < lid_threshold:
                    rejected["language_id_not_tajik_or_low_confidence"] += 1
                    continue

            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                rejected["exact_duplicate"] += 1
                continue
            seen.add(digest)
            source = str(row.get("source") or input_path.stem)
            doc_id = str(row.get("id") or digest)
            split = stable_split(doc_id, seed, validation_percent, test_percent)
            splits[split].append({"text": text, "source": source, "doc_id": doc_id})
            aggregates["letters"] += int(stats["letters"])
            aggregates["cyrillic_letters"] += int(stats["cyrillic"])
            aggregates["arabic_letters"] += int(stats["arabic"])
            aggregates["latin_letters"] += int(stats["latin"])
            aggregates["tajik_marker_letters"] += int(stats["tajik_marker_letters"])
            source_counts[source] += 1

    total_kept = sum(len(items) for items in splits.values())
    if total_kept == 0:
        raise ValueError("No documents passed the filters; inspect the source schema and thresholds")

    for name, rows in splits.items():
        rows.sort(key=lambda item: item["doc_id"])
        write_jsonl(output_dir / f"{name}.jsonl", rows)

    total_letters = max(1, int(aggregates["letters"]))
    manifest = {
        "schema_version": 1,
        "target_language": "tg",
        "script": "Cyrillic",
        "seed": seed,
        "filters": {
            "text_fields": list(text_fields),
            "language_fields": list(lang_fields),
            "assume_language": assume_language,
            "lid_model_path": str(lid_model_path) if lid_model_path else None,
            "lid_threshold": lid_threshold,
            "min_alpha_chars": min_alpha_chars,
            "min_cyrillic_ratio": min_cyrillic_ratio,
            "max_arabic_ratio": max_arabic_ratio,
            "max_latin_ratio": max_latin_ratio,
            "validation_percent": validation_percent,
            "test_percent": test_percent
        },
        "counts": {
            "records_seen": aggregates["records_seen"],
            "documents_kept": total_kept,
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
            "rejected": dict(rejected)
        },
        "script_statistics": {
            "cyrillic_ratio": aggregates["cyrillic_letters"] / total_letters,
            "arabic_ratio": aggregates["arabic_letters"] / total_letters,
            "latin_ratio": aggregates["latin_letters"] / total_letters,
            "tajik_marker_letters": aggregates["tajik_marker_letters"]
        },
        "source_counts": dict(source_counts),
        "language_identification": {
            "model": str(lid_model_path) if lid_model_path else None,
            "documents_checked": lid_checked,
            "mean_top1_confidence": lid_confidence_sum / lid_checked if lid_checked else None
        },
        "limitations": [
            "Script ratios do not distinguish Tajik from Russian or other Cyrillic languages.",
            "Exact duplicate removal does not catch near-duplicates.",
            "Dataset-level language assertions require manual source sampling.",
            "Optional fastText language identification is a screening signal, not a proof of language quality."
        ]
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-fields", default=",".join(DEFAULT_TEXT_FIELDS))
    parser.add_argument("--lang-fields", default=",".join(DEFAULT_LANG_FIELDS))
    parser.add_argument("--assume-language", help="Explicit dataset-level language assertion, e.g. tg")
    parser.add_argument("--min-alpha-chars", type=int, default=80)
    parser.add_argument("--min-cyrillic-ratio", type=float, default=0.55)
    parser.add_argument("--max-arabic-ratio", type=float, default=0.12)
    parser.add_argument("--max-latin-ratio", type=float, default=0.35)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--test-percent", type=int, default=5)
    parser.add_argument("--seed", default="tajik-v1")
    parser.add_argument("--lid-model", type=Path, help="Optional fastText lid.176.ftz model file")
    parser.add_argument("--lid-threshold", type=float, default=0.70)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        manifest = prepare(
            inputs=args.inputs,
            output_dir=args.output_dir,
            text_fields=tuple(x.strip() for x in args.text_fields.split(",") if x.strip()),
            lang_fields=tuple(x.strip() for x in args.lang_fields.split(",") if x.strip()),
            assume_language=args.assume_language,
            min_alpha_chars=args.min_alpha_chars,
            min_cyrillic_ratio=args.min_cyrillic_ratio,
            max_arabic_ratio=args.max_arabic_ratio,
            max_latin_ratio=args.max_latin_ratio,
            validation_percent=args.validation_percent,
            test_percent=args.test_percent,
            seed=args.seed,
            overwrite=args.overwrite,
            lid_model_path=args.lid_model,
            lid_threshold=args.lid_threshold,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
