# Selective Retrieval for Single-Turn Mental-Health QA

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

- MentalChat16K for generator fine-tuning
- CounselBench-Eval and CounselBench-Adv for evaluation

Raw guideline documents, model checkpoints, API keys, and full generated response files are not redistributed in this repository due to size and licensing constraints.

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
