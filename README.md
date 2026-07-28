# Selective Retrieval for Single-Turn Mental-Health QA

## Selective Gate Configuration

The selective gate first applies a fixed, regex-based hard-safety detector to
the user query. For non-safety queries, the same MentalChat16K-tuned Gemma
model used for response generation greedily scores the user query together
with its closed-book draft (`do_sample=False`). It produces integer
information, coping, and specificity need scores on a 1–5 scale.

Hard safety always retrieves from the safety corpus. Otherwise, retrieval is
activated when either the mean of the three scores is at least `3.25` or the
larger of the information and coping scores is at least `4.0`. A high-axis
activation routes to `coping` when the coping score is greater than or equal
to the information score, and to `psychoeducation` otherwise. Activation by
the mean threshold alone routes to `all_non_safety`.

This repository contains the implementation for the KDD Undergraduate Consortium submission:

**When Retrieval Helps: Selective Retrieval for Single-Turn Mental-Health QA**

## Overview

This project studies whether retrieval should be applied selectively in single-turn mental-health question answering. The system compares:

- closed-book generation,
- always-retrieval generation,
- selective retrieval with a hard safety trigger and a soft retrieval-utility gate.

## Main Components

- `src/gate.py`: selective retrieval policy
- `src/retriever.py`: BM25 retrieval over the guideline corpus
- `src/generator.py`: generator loading and response generation
- `scripts/generate_responses.py`: response generation for no-retrieval, always-retrieval, and selective-retrieval settings
- `scripts/run_llm_judge_eval.py`: CounselBench-Eval judging
- `scripts/run_llm_judge_adv.py`: CounselBench-Adv judging
- `scripts/aggregate_eval_scores.py`: aggregation for CounselBench-Eval results
- `scripts/aggregate_adv_scores.py`: aggregation for CounselBench-Adv results
- `scripts/plot_calibration.py`: threshold calibration plots
## Data

The experiments use public benchmark datasets:

- MentalChat16K for generator fine-tuning: https://huggingface.co/datasets/ShenLab/MentalChat16K
- CounselBench-Eval and CounselBench-Adv for evaluation: https://github.com/llm-eval-mental-health/CounselBench

The guideline corpus is constructed from public mental-health resources following the procedure described in the paper and code. Raw guideline documents, model checkpoints, API keys, and full generated response files are not redistributed in this repository due to size, licensing, and safety considerations.

## Reproduction Sketch

Create the environment:

    conda env create -f environment.yml
    conda activate mh-rag

Generate responses:

    python scripts/generate_responses.py --mode no_retrieval ...
    python scripts/generate_responses.py --mode always_retrieval ...
    python scripts/generate_responses.py --mode gated_retrieval ...

Run LLM judges:

    python scripts/run_llm_judge_eval.py ...
    python scripts/run_llm_judge_adv.py ...

Aggregate results:

    python scripts/aggregate_eval_scores.py ...
    python scripts/aggregate_adv_scores.py ...

## Notes

This repository is intended to support reproducibility of the reported retrieval-policy experiments. Some artifacts are omitted because they are large, private, or dependent on external API credentials.
