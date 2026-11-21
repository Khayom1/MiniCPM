#!/usr/bin/env python3
"""Continued pretraining of a causal LM on a prepared Tajik text corpus."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def arguments():
    p = argparse.ArgumentParser(description="TAJIK-CORE continued pretraining")
    p.add_argument("--model", required=True)
    p.add_argument("--train-file", required=True, type=Path)
    p.add_argument("--validation-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()

def main():
    a = arguments()
    if a.bf16 and a.fp16: raise SystemExit("Choose only one of --bf16 or --fp16.")
    if a.max_length < 32 or a.batch_size < 1 or a.gradient_accumulation < 1: raise SystemExit("Invalid training dimensions.")
    for f in (a.train_file, a.validation_file):
        if not f.is_file(): raise SystemExit(f"Dataset file not found: {f}")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None: raise SystemExit("Tokenizer requires eos_token_id or pad_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if a.bf16 else torch.float16 if a.fp16 else None
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, trust_remote_code=a.trust_remote_code)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    if a.use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc: raise SystemExit("Install peft to enable LoRA.") from exc
        linear_names = {n.rsplit(".", 1)[-1] for n, m in model.named_modules()
                        if isinstance(m, torch.nn.Linear) and not n.endswith(("lm_head", "embed_tokens"))}
        candidates = ("q_proj", "k_proj", "v_proj", "o_proj", "q_a_proj", "q_b_proj",
                      "kv_a_proj_with_mqa", "kv_b_proj", "wq", "wk", "wv", "wo")
        targets = sorted(set(candidates) & linear_names)
        if not targets: raise SystemExit(f"No supported attention projections found for LoRA. Linear modules: {sorted(linear_names)[:80]}")
        model = get_peft_model(model, LoraConfig(r=a.lora_rank, lora_alpha=a.lora_alpha,
            lora_dropout=a.lora_dropout, target_modules=targets, bias="none", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    raw = load_dataset("json", data_files={"train": str(a.train_file), "validation": str(a.validation_file)})
    for split in ("train", "validation"):
        if "text" not in raw[split].column_names: raise SystemExit(f"{split} dataset requires a text field.")
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=a.max_length, padding=False, add_special_tokens=True)
    tokenized = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names, desc="Tokenizing Tajik corpus")
    tokenized = tokenized.filter(lambda row: len(row["input_ids"]) >= 2, desc="Removing empty records")
    if not len(tokenized["train"]) or not len(tokenized["validation"]):
        raise SystemExit("Both train and validation splits must contain tokenized documents.")
    kwargs = dict(output_dir=str(a.output_dir), num_train_epochs=a.epochs, learning_rate=a.learning_rate,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.gradient_accumulation, warmup_ratio=a.warmup_ratio,
        weight_decay=a.weight_decay, logging_steps=a.logging_steps, save_steps=a.save_steps,
        save_strategy="steps", eval_steps=a.save_steps, report_to="none", seed=a.seed,
        bf16=a.bf16, fp16=a.fp16, gradient_checkpointing=a.gradient_checkpointing,
        remove_unused_columns=False, save_total_limit=3)
    if "eval_strategy" in TrainingArguments.__dataclass_fields__: kwargs["eval_strategy"] = "steps"
    else: kwargs["evaluation_strategy"] = "steps"
    trainer = Trainer(model=model, args=TrainingArguments(**kwargs), train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer)
    trainer.train()
    final_dir = a.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics = trainer.evaluate()
    if "eval_loss" in metrics: metrics["perplexity"] = float(torch.exp(torch.tensor(metrics["eval_loss"])))
    (final_dir / "training_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
