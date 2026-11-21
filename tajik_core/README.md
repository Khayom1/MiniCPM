# TAJIK-CORE

Tajik-language continued-pretraining pipeline inside the MiniCPM repository.

## What this is

This package prepares a deduplicated Tajik Cyrillic corpus and trains a causal language model on that corpus, optionally with LoRA. It is not a pretrained Tajik checkpoint. A trained checkpoint requires a sufficiently large, legally usable corpus and suitable accelerator hardware. Script completion or a small sample corpus is not evidence of native-level Tajik proficiency.

## Prepare the corpus

Accepted source formats are UTF-8 TXT, JSON, and JSONL. JSON records may contain `text`, ChatML `messages`, or instruction/input/output fields.

    python tajik_core/prepare_corpus.py --input /path/to/tajik_sources --output-dir data/tajik_core --min-chars 40 --validation-ratio 0.02

The pipeline applies Unicode NFC normalization, whitespace cleanup, a coarse Cyrillic-dominance filter, exact deduplication, and deterministic train/validation splitting. It does not invent, translate, or rewrite source text. Review provenance and licensing before use. The Cyrillic ratio is not a full Tajik language detector; human or independently validated language review is required.

## Install

Use a Linux host with a supported NVIDIA GPU and CUDA-compatible PyTorch. Android/Termux is suitable for collection and light preprocessing, not for training an 8B-class model.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r tajik_core/requirements.txt

## Continued pretraining

    python tajik_core/train.py --model openbmb/MiniCPM4-8B --train-file data/tajik_core/train.jsonl --validation-file data/tajik_core/validation.jsonl --output-dir outputs/tajik-core --max-length 2048 --epochs 1 --bf16 --gradient-checkpointing --trust-remote-code

LoRA variant: add `--use-lora`. Confirm the model identifier, license, tokenizer, architecture compatibility, and hardware requirements before launching. The checkpoint is saved under `outputs/tajik-core/final`.

## Quality requirements

Keep literary Tajik, colloquial Tajik, regional speech, scientific/medical texts, and technical material traceable as separate source collections so their proportions can be controlled. Exclude duplicated boilerplate, OCR corruption, unreviewed machine translation, and material without appropriate usage rights. Keep an independently authored evaluation corpus out of training.

A model cannot be described as “100% fluent” from training completion or perplexity alone. Such a claim requires broad evaluation of grammar, semantics, idioms, dialects, long-context comprehension, factuality, translation, and instruction following against a comparable English benchmark.
