import os
import argparse
import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="google/gemma-4-E4B-it")
    p.add_argument("--output_dir", type=str, default="./outputs/gemma4-e4b-it-mh16k")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None)

    # checkpoint / resume 관련
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    return p.parse_args()


def load_mh_dataset(tokenizer, max_samples=None):
    raw = load_dataset("ShenLab/MentalChat16K")

    if isinstance(raw, DatasetDict):
        if "train" in raw:
            ds = raw["train"]
        else:
            ds = concatenate_datasets([raw[k] for k in raw.keys()])
    else:
        ds = raw

    ds = ds.filter(
        lambda x: x.get("instruction") is not None
        and x.get("input") is not None
        and x.get("output") is not None
    )

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def to_text(example):
        messages = [
            {"role": "system", "content": example["instruction"].strip()},
            {"role": "user", "content": example["input"].strip()},
            {"role": "assistant", "content": example["output"].strip()},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    return ds


def main():
    args = get_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 안 잡힘.")

    os.makedirs(args.output_dir, exist_ok=True)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[INFO] compute_dtype = {compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[INFO] Loading dataset...")
    ds = load_mh_dataset(tokenizer, max_samples=args.max_samples)
    splits = ds.train_test_split(test_size=0.02, seed=42)
    train_ds = splits["train"]
    eval_ds = splits["test"]

    print(f"[INFO] train size = {len(train_ds)}")
    print(f"[INFO] eval size  = {len(eval_ds)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
    )

    print("[INFO] Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto",
        dtype=compute_dtype,
        quantization_config=bnb_config,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,

        # 핵심: epoch 말고 steps로 저장
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,

        # eval도 steps 단위로 맞춤
        eval_strategy="steps",
        eval_steps=args.save_steps,

        optim="adamw_torch_fused",
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="text",
        packing=False,

    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # resume 우선순위:
    # 1) 사용자가 직접 --resume_from_checkpoint 지정
    # 2) output_dir에서 마지막 checkpoint 자동 탐지
    resume_ckpt = None
    if args.resume_from_checkpoint is not None:
        resume_ckpt = args.resume_from_checkpoint
        print(f"[INFO] Using user-provided checkpoint: {resume_ckpt}")
    else:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            resume_ckpt = last_checkpoint
            print(f"[INFO] Auto-detected checkpoint: {resume_ckpt}")
        else:
            print("[INFO] No checkpoint found. Starting from scratch.")

    print("[INFO] Start training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print("[INFO] Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
