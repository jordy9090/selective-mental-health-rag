#!/usr/bin/env python3
"""Publish the KDD QLoRA adapter and model card to Hugging Face Hub.

This script uploads adapter-only PEFT artifacts. It does not load or upload the
base Gemma model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "mira2020/gemma-4-e4b-mentalchat16k-qlora"
DEFAULT_MODEL_CARD = (
    Path(__file__).resolve().parent.parent / "hf_model_card" / "README.md"
)

REQUIRED_CONFIG = "adapter_config.json"
WEIGHT_PATTERNS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model-*.safetensors",
    "adapter_model-*.bin",
)
OPTIONAL_FILES = (
    "adapter_model.safetensors.index.json",
    "adapter_model.bin.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a PEFT/QLoRA adapter and model card to Hugging Face Hub."
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Directory containing adapter_config.json and adapter weights.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Target Hugging Face model repository (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=DEFAULT_MODEL_CARD,
        help="Markdown file to upload as README.md.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face repository as private. Omit for a public repo.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Prefer cached `hf auth login` or HF_TOKEN.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the upload plan without creating or uploading anything.",
    )
    return parser.parse_args()


def collect_adapter_files(adapter_dir: Path) -> list[Path]:
    adapter_dir = adapter_dir.expanduser().resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    config_path = adapter_dir / REQUIRED_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing {REQUIRED_CONFIG} in adapter directory: {adapter_dir}"
        )

    weight_files: list[Path] = []
    for pattern in WEIGHT_PATTERNS:
        weight_files.extend(sorted(adapter_dir.glob(pattern)))
    weight_files = list(dict.fromkeys(weight_files))

    if not weight_files:
        raise FileNotFoundError(
            "No PEFT adapter weights found. Expected adapter_model.safetensors, "
            "adapter_model.bin, or sharded adapter_model-* files."
        )

    files = [config_path, *weight_files]
    files.extend(
        path for name in OPTIONAL_FILES if (path := adapter_dir / name).is_file()
    )
    return list(dict.fromkeys(files))


def main() -> int:
    args = parse_args()

    try:
        adapter_files = collect_adapter_files(args.adapter_dir)
        model_card = args.model_card.expanduser().resolve()
        if not model_card.is_file():
            raise FileNotFoundError(f"Model card does not exist: {model_card}")
    except (FileNotFoundError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(f"Target repository: https://huggingface.co/{args.repo_id}")
    print(f"Visibility: {'private' if args.private else 'public'}")
    print("Files to upload:")
    for path in adapter_files:
        print(f"  - {path} -> {path.name}")
    print(f"  - {model_card} -> README.md")

    if args.dry_run:
        print("Dry run complete; nothing was uploaded.")
        return 0

    token = args.token or os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    try:
        identity = api.whoami()
        print(f"Authenticated as: {identity.get('name', 'unknown')}")

        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )

        for path in adapter_files:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Upload {path.name}",
            )

        api.upload_file(
            path_or_fileobj=str(model_card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
    except Exception as exc:  # Hugging Face client exposes several HTTP error types.
        print(f"[ERROR] Upload failed: {exc}", file=sys.stderr)
        return 1

    print(f"Published: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
