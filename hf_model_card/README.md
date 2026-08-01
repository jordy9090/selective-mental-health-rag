---
language:
- en
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
base_model: google/gemma-4-E4B-it
datasets:
- ShenLab/MentalChat16K
tags:
- peft
- lora
- qlora
- gemma4
- mental-health
- question-answering
- retrieval-augmented-generation
---

# Gemma-4-E4B-it MentalChat16K QLoRA Adapter

This repository contains the PEFT/QLoRA adapter used in **“When Retrieval Helps: Selective Retrieval for Single-Turn Mental-Health QA”** (KDD Undergraduate Consortium 2026).

The adapter domain-adapts `google/gemma-4-E4B-it` on `ShenLab/MentalChat16K`. In the paper, the resulting model is held fixed as the shared generator across Closed-book, Always Retrieval, and Selective Retrieval settings so that retrieval-policy effects can be compared under the same generator.

This is an **adapter-only release**. The base model weights are not redistributed.

## Intended Use

The adapter is intended for research on:

- single-turn mental-health question answering,
- retrieval-augmented generation,
- selective retrieval and retrieval routing,
- safety-sensitive response evaluation.

It is not a clinical system and must not be used for diagnosis, treatment planning, medication guidance, crisis intervention, or decisions affecting access to healthcare.

## Load the Adapter

```python
import torch
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)

BASE_MODEL = "google/gemma-4-E4B-it"
ADAPTER = "mira2020/gemma-4-e4b-mentalchat16k-qlora"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
)

model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()
```

The LoRA configuration is stored in `adapter_config.json`. Loading the base model may require accepting its license terms on Hugging Face.

## Training Data

The adapter was trained on `ShenLab/MentalChat16K`, an English single-turn conversational mental-health assistance dataset containing synthetic counseling question-answer pairs and anonymized intervention transcripts.

No MentalChat16K examples are redistributed in this model repository.

## Training Procedure

- Base model: `google/gemma-4-E4B-it`
- Training dataset: `ShenLab/MentalChat16K`
- Adaptation method: QLoRA with PEFT LoRA
- Quantization: 4-bit NF4 with double quantization
- LoRA rank: 64
- LoRA alpha: 16
- LoRA dropout: 0.1
- Target modules: all linear layers
- Epochs: 3
- Learning rate: `5e-5`
- Maximum sequence length: 1024
- Per-device batch size: 1
- Gradient accumulation steps: 8
- Optimizer: fused AdamW
- Learning-rate schedule: constant
- Maximum gradient norm: 0.3
- Checkpoint and evaluation interval: 100 steps

The released `adapter_config.json` records the adapter architecture. The accompanying code repository documents the training, inference, and selective-retrieval pipeline.

## Evaluation Context

This adapter served as the shared generator in the paper’s controlled retrieval-policy comparison. The following are system-level results on CounselBench; they should not be interpreted as standalone clinical validation of the adapter.

### CounselBench-Eval

| System | Overall ↑ | Empathy ↑ | Specificity ↑ | Medical Advice ↓ | Retrieval Rate |
|---|---:|---:|---:|---:|---:|
| Tuned Closed-book | 4.15 | 4.81 | 3.92 | 0.00 | 0.0% |
| Tuned + Always Retrieval | 4.12 | 4.78 | 3.97 | 0.01 | 100.0% |
| Tuned + Selective Retrieval | 4.17 | 4.83 | 3.96 | 0.00 | 9.0% |

### CounselBench-Adv

| System | Macro Failure ↓ | Retrieval Rate |
|---|---:|---:|
| Tuned Closed-book | 0.0250 | 0.0% |
| Tuned + Always Retrieval | 0.0917 | 100.0% |
| Tuned + Selective Retrieval | 0.0250 | 7.5% |

The main finding is that unconditional retrieval can improve specificity while introducing additional safety-sensitive failures. Selective retrieval limits this degradation by preserving closed-book responses for low-need cases.

## Limitations

- The adapter was evaluated with one base-model family.
- Evaluation uses two splits from the same benchmark framework.
- The paper’s retrieval experiments use a compact, controlled guideline corpus.
- Evaluation relies primarily on LLM judges and a small targeted expert audit.
- The model is designed for single-turn QA and does not represent longitudinal therapeutic context.

The model may generate incorrect, overly clinical, directive, or otherwise inappropriate content. Human review is required for any safety-sensitive use.

## Ethical and Safety Considerations

This release is for research and reproducibility. It does not replace qualified mental-health professionals. Do not rely on the model for emergencies, diagnosis, medication decisions, or individualized treatment. Applications should implement independent safety checks, local crisis-resource handling, privacy protection, and qualified human oversight.

## Code

Implementation and evaluation code:

- https://github.com/jordy9090/selective-mental-health-rag

## Citation

```bibtex
@inproceedings{oh2026when,
  title     = {When Retrieval Helps: Selective Retrieval for Single-Turn Mental-Health QA},
  author    = {Oh, Hyunseo and Kim, Chong-Kwon and Choi, Yoonhyuk},
  booktitle = {KDD Undergraduate Consortium},
  year      = {2026}
}
```
