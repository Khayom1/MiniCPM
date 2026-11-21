# Tajik language adaptation for MiniCPM5

This directory is the reproducible data and evaluation foundation for adapting MiniCPM5 to Tajik (тоҷикӣ), with Tajik Cyrillic as the primary script.

## Important scope

A tokenizer accepting Cyrillic bytes does **not** prove that a model understands Tajik. The tokenizer audit below measures round-trip fidelity and token efficiency; language competence must be established separately with held-out tests.

There is no technically defensible way to promise "100% understanding" for any natural language model. The engineering target is measurable competence across spelling, morphology, syntax, semantics, discourse, domain knowledge, instruction following, and the Dushanbe conversational register.

## Model and training format

The upstream MiniCPM5-2B repository documents a standard `LlamaForCausalLM` architecture and an official MiniCPM5 chat format. For SFT, use LLaMA-Factory's **`template: minicpm5`**. Do not replace it with `empty`, `llama3`, or another generic template.

Recommended first experiment:
- Base: `openbmb/MiniCPM5-2B` (the instruction-tuned release).
- Method: LoRA SFT on reviewed Tajik instruction/answer examples.
- Data: JSONL, one `{"messages":[...]}` record per line.
- Hardware: a supported CUDA GPU host. The Android phone is for inference and data inspection, not 2B model training.

For deeper language adaptation, a separately measured continued-pretraining experiment can be evaluated from an appropriate base/midtrain checkpoint. Do not start that run until the corpus license, domain balance, tokenizer statistics, and a held-out baseline are verified. Continued pretraining and instruction SFT are different objectives; mixing raw documents into chat SFT is incorrect.

## Public data candidates and licensing gates

See `data_sources.json` for the machine-readable inventory.

- `TajikNLPWorld/tajik-web-corpus`: large text source, but its card says original source texts remain owned by their authors/publishers. The compilation license does not relicense the underlying text. Do not redistribute it in this repository; verify the usage rights and remove Persian/non-Tajik contamination before training.
- `TajikNLPWorld/TajPersParallelCorpusFull`: Apache-2.0; useful for Tajik–Persian alignment, lexical coverage and transliteration/translation tasks. Its contents include poetry, prose and lexical pairs, so it is not a balanced conversational corpus.
- `TajikNLPWorld/TajikUnifiedCorpus`: Apache-2.0 morphological resource, but its Hub card currently gates file access behind an access/consent step. Use only after the access terms are reviewed.
- The test/evaluation split must not be used for training, prompt examples, or synthetic-data generation.

Dataset cards and conditions can change. Re-check the source card and underlying rights before downloading, training, or publishing any derived artifacts.

## Pipeline

All scripts use UTF-8 and the Python standard library except the optional Hugging Face tokenizer audit.

### 1. Prepare raw text documents

Export source records as JSONL. Each record must contain one text field (`text`, `tajik`, `content`, `body`, or `document`) and, when available, a language tag (`lang` or `language`).

For stronger filtering, install the optional fastText dependency and download the compressed LID-176 model (917 KB):

```bash
pip install fasttext-wheel
curl -L -o lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

LID-176 supports the ISO language label `tg`. It was trained on Wikipedia, Tatoeba and SETimes data and is licensed CC-BY-SA. Keep its license notice. It is a screening signal, not a language-quality guarantee.

For a source whose dataset-level card declares Tajik but whose rows have no language field, the caller must explicitly opt in:

```bash
python tajik/prepare_corpus.py \
  --inputs /path/to/source.jsonl \
  --output-dir /path/to/prepared-tajik \
  --assume-language tg \
  --lid-model ./lid.176.ftz \
  --lid-threshold 0.70
```

This option is an explicit assertion about the dataset, not a language detector. The optional LID model rejects texts whose top prediction is not `tg` or whose confidence is below the configured threshold. Script-ratio checks also reject obvious script contamination. Neither method guarantees Tajik: manually review samples from every source and inspect borderline cases before accepting a corpus.

Output:
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: deterministic document-level splits with `text`, `source`, and `doc_id`.
- `manifest.json`: counts, filtering settings and script statistics.

The split is hash-based and reproducible. It removes exact duplicates, not semantic/near duplicates. Use a near-duplicate audit for large web corpora before training.

### 2. Audit the tokenizer

Install `transformers` in a separate environment, then run:

```bash
pip install transformers
python tajik/audit_tokenizer.py \
  --model openbmb/MiniCPM5-2B \
  --input /path/to/prepared-tajik/train.jsonl
```

The report includes Cyrillic/Arabic/Latin script proportions, tokens per character, unknown-token count, and exact decode round-trip rate. A byte-level tokenizer can represent Unicode text while still splitting a language inefficiently; the measurements matter.

### 3. Validate the SFT examples

Create a reviewed `messages` dataset; never fabricate an answer merely to turn a raw document into a Q&A pair.

```bash
python tajik/validate_sft.py --inputs /path/to/tajik_sft.jsonl
```

The expected format is one JSON object per line:

```json
{"messages":[{"role":"user","content":"Салом. Ин ҷумла чӣ маъно дорад?"},{"role":"assistant","content":"Ин ҷумла ..."}]}
```

### 4. LoRA SFT

Install LLaMA-Factory from source as required by the upstream MiniCPM5 guide. Copy `configs/dataset_info.json` and `configs/lora_sft.yaml` into the training data/config location, adjust the absolute paths, and run on a supported GPU machine. The correct template is `minicpm5`.

The Galaxy A35 is not a realistic training host for this 2B model. Use it for collection review, validation, and inference after the adapter has been merged/converted. Do not assume a cloud GPU is free or available until its quota and actual device are checked.

## Evaluation gates

Do not declare the language adaptation complete from training loss alone. Maintain a private, human-reviewed test set covering:
1. Tajik orthography and the letters ғ, ӣ, қ, ӯ, ҳ, ҷ.
2. Morphology: verb tense/aspect/person, noun number/case-like forms, agreement, derivation and clitics.
3. Syntax, word order, negation, questions, pronouns and complex sentences.
4. Semantic understanding, paraphrase, summarization and multi-turn reference.
5. Tajik ↔ Russian translation, with Persian kept distinct from Tajik.
6. General knowledge, factuality and uncertainty in Tajik.
7. Formal Tajik, educational Tajik, news register and colloquial Dushanbe Tajik.
8. Robustness to spelling variants, typos, transliteration and code-switching.

Report each dimension separately and compare the untouched base model with the adapted model. A higher score on one subset must not hide regressions elsewhere.

## Current implementation status

This branch establishes a safe corpus-preparation and evaluation foundation. It does not contain a trained adapter or model weights, and it does not claim that the model already understands Tajik. Training starts only after data rights, data quality, tokenizer audit, and baseline evaluation have passed.
