"""
Upload LoRA Adapters to HuggingFace Hub
========================================

Upload SFT and GRPO LoRA adapters to the same HuggingFace repository,
organized into subdirectories by training stage.

Target repository structure:
    your-username/humor-qwen3-8b-lora/
    ├── sft/                  ← SFT LoRA adapter
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── ...
    ├── grpo/                 ← GRPO LoRA adapter
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── ...
    └── README.md

Usage:
    # First-time login
    huggingface-cli login

    # Reorganize existing repo (move root-level SFT files into sft/ subdirectory)
    python -m utils.upload_model --reorganize --repo_id YOUR_USERNAME/humor-qwen3-8b-lora

    # Upload SFT adapter (to sft/ subdirectory)
    python -m utils.upload_model --stage sft --repo_id YOUR_USERNAME/humor-qwen3-8b-lora

    # Upload GRPO adapter (to grpo/ subdirectory)
    python -m utils.upload_model --stage grpo --repo_id YOUR_USERNAME/humor-qwen3-8b-lora

Dependencies:
    pip install huggingface_hub
"""

import argparse
from pathlib import Path

from huggingface_hub import (
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
)


# ============================================================
# Path Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default local adapter paths for each training stage
STAGE_DEFAULTS = {
    "sft": {
        "adapter_path": PROJECT_ROOT / "checkpoints" / "sft" / "final",
        "path_in_repo": "sft",
    },
    "grpo": {
        "adapter_path": PROJECT_ROOT / "checkpoints" / "grpo" / "final",
        "path_in_repo": "grpo",
    },
}


# ============================================================
# Reorganize: Move Root-Level Files into sft/ Subdirectory
# ============================================================

def reorganize_repo(api: HfApi, repo_id: str) -> None:
    """Move existing root-level SFT adapter files into sft/ subdirectory.

    When the SFT adapter was originally uploaded to the repo root (before
    the multi-stage directory structure was adopted), this function
    reorganizes the repo by moving those files into a sft/ subdirectory.

    Uses CommitOperationCopy + CommitOperationDelete in a single atomic
    commit: files are copied to their new location and originals are
    deleted, so there is no intermediate state where files are missing.

    Files already inside subdirectories (e.g., sft/, grpo/) or special
    files (README.md, .gitattributes) are left untouched.

    Args:
        api: Authenticated HfApi instance.
        repo_id: HuggingFace repository ID (e.g., "user/repo-name").
    """
    all_files = api.list_repo_files(repo_id=repo_id)

    # Identify root-level files that should be moved into sft/
    # Skip: files already in subdirectories, README.md, .gitattributes
    skip_prefixes = ("sft/", "grpo/", ".")
    skip_names = {"README.md", ".gitattributes"}

    files_to_move = [
        f for f in all_files
        if not any(f.startswith(p) for p in skip_prefixes)
        and f not in skip_names
    ]

    if not files_to_move:
        print("  No root-level files to reorganize. Repo structure is already clean.")
        return

    print(f"  Found {len(files_to_move)} root-level files to move into sft/:")
    for f in files_to_move:
        print(f"    {f} → sft/{f}")

    # Build atomic commit: copy to new location + delete original
    operations = []
    for f in files_to_move:
        operations.append(CommitOperationCopy(
            src_path_in_repo=f,
            path_in_repo=f"sft/{f}",
        ))
        operations.append(CommitOperationDelete(path_in_repo=f))

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="Reorganize: move SFT adapter files into sft/ subdirectory",
    )

    print(f"  Reorganization complete. {len(files_to_move)} files moved to sft/.")


# ============================================================
# Upload Stage Adapter
# ============================================================

def upload_stage(
    api: HfApi,
    repo_id: str,
    stage: str,
    adapter_path: Path,
) -> None:
    """Upload a LoRA adapter to the corresponding subdirectory in the repo.

    Args:
        api: Authenticated HfApi instance.
        repo_id: HuggingFace repository ID.
        stage: Training stage name ("sft" or "grpo").
        adapter_path: Local path to the adapter directory.
    """
    path_in_repo = STAGE_DEFAULTS[stage]["path_in_repo"]

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter path does not exist: {adapter_path}"
        )

    print(f"  Uploading {adapter_path} → {repo_id}/{path_in_repo}/")

    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        commit_message=f"Upload {stage.upper()} LoRA adapter",
    )

    print(f"  Upload complete: https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Upload LoRA adapters to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_id", type=str, required=True,
        help="HuggingFace repo ID (e.g., your-username/humor-qwen3-8b-lora)",
    )
    parser.add_argument(
        "--stage", type=str, choices=["sft", "grpo"], default=None,
        help="Training stage to upload: 'sft' or 'grpo'",
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="Override default adapter path for the stage",
    )
    parser.add_argument(
        "--reorganize", action="store_true",
        help="Reorganize existing repo: move root-level SFT files "
             "into sft/ subdirectory. Run this once if you previously "
             "uploaded SFT adapter to the repo root.",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create repo as private (only applies when creating new repo)",
    )
    args = parser.parse_args()

    if not args.stage and not args.reorganize:
        parser.error("Either --stage or --reorganize is required.")

    api = HfApi()

    # Ensure repo exists
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private)

    # Reorganize if requested
    if args.reorganize:
        print("=" * 60)
        print(f"Reorganizing repo: {args.repo_id}")
        print("=" * 60)
        reorganize_repo(api, args.repo_id)

    # Upload stage adapter if requested
    if args.stage:
        adapter_path = (
            Path(args.adapter_path)
            if args.adapter_path
            else STAGE_DEFAULTS[args.stage]["adapter_path"]
        )

        print("=" * 60)
        print(f"Uploading {args.stage.upper()} adapter")
        print("=" * 60)
        upload_stage(api, args.repo_id, args.stage, adapter_path)

    print(f"\nDone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
