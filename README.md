# Selective Retrieval for Single-Turn Mental-Health QA

This repository contains the implementation for the KDD Undergraduate Consortium 2026 paper:

**When Retrieval Helps: Selective Retrieval for Single-Turn Mental-Health QA**

## Overview

This project studies whether retrieval should be applied selectively in single-turn mental-health question answering. The system compares:

- closed-book generation,
- always-retrieval generation,
- selective retrieval with a hard safety trigger and a soft retrieval-utility gate.

## Hugging Face Adapter

The MentalChat16K QLoRA adapter used as the shared generator is published at:

- https://huggingface.co/mira2020/gemma-4-e4b-mentalchat16k-qlora

The repository contains adapter-only PEFT weights for `google/gemma-4-E4B-it`; it does not redistribute the base-model weights.

The Hugging Face model card source is stored at `hf_model_card/README.md`.

### Publish or Update the Adapter

Authenticate once on the machine containing the final adapter checkpoint:

```bash
hf auth login
hf auth whoami
```

Validate the upload without changing the Hub:

```bash
python scripts/publish_adapter_to_hf.py \
  --adapter-dir /absolute/path/to/final_adapter \
  --dry-run
```

Publish the public adapter and model card:

```bash
python scripts/publish_adapter_to_hf.py \
  --adapter-dir /absolute/path/to/final_adapter
```

The script validates `adapter_config.json` and adapter weight files, creates the public model repository if needed, uploads adapter-only artifacts, and installs `hf_model_card/README.md` as the Hub model card.

### Camera-ready Appendix Line

```latex
\noindent\textbf{Model.} The MentalChat16K-adapted QLoRA adapter is publicly available at \url{https://huggingface.co/mira2020/gemma-4-e4b-mentalchat16k-qlora}.
```

## Selective Gate Configuration

The selective gate first applies a fixed, regex-based hard-safety detector to the user query. For non-safety queries, the same MentalChat16K-tuned Gemma model used for response generation greedily scores the user query together with its closed-book draft (`do_sample=False`). It produces integer information, coping, and specificity need scores on a 1–5 scale.

Hard safety always retrieves from the safety corpus. Otherwise, retrieval is activated when either the mean of the three scores is at least `3.25` or the larger of the information and coping scores is at least `4.0`. A high-axis activation routes to `coping` when the coping score is greater than or equal to the information score, and to `psychoeducation` otherwise. Activation by the mean threshold alone routes to `all_non_safety`.

## Main Components

- `src/gate.py`: selective retrieval policy
- `src/retriever.py`: BM25 retrieval over the guideline corpus
- `src/generator.py`: generator loading and response generation
- `scripts/generate_responses.py`: response generation for closed-book, always-retrieval, and selective-retrieval settings
- `scripts/run_llm_judge_eval.py`: CounselBench-Eval judging
- `scripts/run_llm_judge_adv.py`: CounselBench-Adv judging
- `scripts/aggregate_eval_scores.py`: aggregation for CounselBench-Eval results
- `scripts/aggregate_adv_scores.py`: aggregation for CounselBench-Adv results
- `scripts/plot_calibration.py`: threshold calibration plots
- `scripts/publish_adapter_to_hf.py`: validated adapter-only Hugging Face publication
- `hf_model_card/README.md`: Hugging Face model card source

## Data

The experiments use public benchmark datasets:

- MentalChat16K for generator fine-tuning: https://huggingface.co/datasets/ShenLab/MentalChat16K
- CounselBench-Eval and CounselBench-Adv for evaluation: https://github.com/llm-eval-mental-health/CounselBench

The guideline corpus is constructed from public mental-health resources following the procedure described in the paper and code. Raw guideline documents, API keys, full generated-response files, and base-model weights are not redistributed in this repository due to licensing, size, and safety considerations.

## Reproduction Sketch

Create the environment:

```bash
conda env create -f environment.yml
conda activate mh-rag
```

Generate responses:

```bash
python scripts/generate_responses.py --mode no_retrieval ...
python scripts/generate_responses.py --mode always_retrieval ...
python scripts/generate_responses.py --mode gated_retrieval ...
```

Run LLM judges:

```bash
python scripts/run_llm_judge_eval.py ...
python scripts/run_llm_judge_adv.py ...
```

Aggregate results:

```bash
python scripts/aggregate_eval_scores.py ...
python scripts/aggregate_adv_scores.py ...
```

## Notes

This repository supports reproducibility of the reported retrieval-policy experiments. The released adapter is a research artifact and is not a clinical system or a substitute for qualified mental-health professionals.
