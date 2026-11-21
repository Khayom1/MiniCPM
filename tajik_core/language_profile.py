#!/usr/bin/env python3
"""Heuristic language profiling for Tajik corpus triage.

This module never claims perfect language identification. It labels documents for
review and deliberately keeps uncertain samples out of automatic deletion paths.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

CYRILLIC = re.compile(r"[\u0400-\u04FF]")
LATIN = re.compile(r"[A-Za-z]")
ARABIC = re.compile(r"[\u0600-\u06FF]")
TAJIK_LETTERS = set("ӣғқӯҳҷӢҒҚӮҲҶ")
# Frequent Tajik function words / forms. These are signals, not a dictionary.
TAJIK_MARKERS = {
    "ва", "дар", "ба", "аз", "бо", "барои", "ки", "ин", "он", "ҳам",
    "нест", "аст", "буд", "шудааст", "мебошад", "мо", "шумо", "онҳо",
    "чӣ", "чаро", "чӣ гуна", "зеро", "агар", "аммо", "вале", "ҳамин",
    "чунин", "дигар", "ҳар", "то", "тоҷикӣ", "забони", "мардум",
}


def profile_text(text: str) -> dict[str, Any]:
    """Return script counts and conservative Tajik-likelihood signals."""
    normalized = unicodedata.normalize("NFC", text)
    letters = [c for c in normalized if c.isalpha()]
    counts = Counter(
        script for c in letters
        for script, pattern in (("cyrillic", CYRILLIC), ("latin", LATIN), ("arabic", ARABIC))
        if pattern.fullmatch(c)
    )
    total = len(letters)
    cyrillic_ratio = counts["cyrillic"] / total if total else 0.0
    tajik_chars = sum(c in TAJIK_LETTERS for c in letters)
    words = re.findall(r"[^\W_]+", normalized.casefold(), flags=re.UNICODE)
    marker_hits = sum(word in TAJIK_MARKERS for word in words)
    marker_ratio = marker_hits / max(1, len(words))
    # Distinctive letters are strong positive evidence, but their absence is
    # not proof that a text is not Tajik.
    score = min(1.0, (tajik_chars / max(1, total)) * 8.0 + marker_ratio * 5.0)
    if total < 20:
        label = "too_short"
    elif cyrillic_ratio < 0.55:
        label = "non_cyrillic_or_mixed"
    elif tajik_chars >= 2 or marker_hits >= 3 or score >= 0.10:
        label = "tajik_candidate"
    else:
        label = "manual_review"
    return {
        "label": label,
        "score": round(score, 4),
        "letters": total,
        "script_counts": dict(counts),
        "cyrillic_ratio": round(cyrillic_ratio, 4),
        "tajik_specific_letters": tajik_chars,
        "tajik_marker_hits": marker_hits,
        "warning": "Heuristic only; validate with human-reviewed samples.",
    }
