from pathlib import Path
from typing import Iterable

import pandas as pd
import wandb


# =========================
# User Configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_PATH = "DNN_Test/huggingface/mj3gmuyb"
METRIC_NAMES = [
    "train/loss",
    "train/reward",
    "train/kl",
    "train/grad_norm",
    "train/rewards/reward_fn/mean",
    "train/rewards/reward_fn/std",
    "train/reward_std",
    "train/entropy",
    "train/completions/min_terminated_length",
    "train/completions/max_terminated_length",
    "train/completions/mean_terminated_length",
    "train/completions/min_length",
    "train/completions/max_length",
    "train/completions/mean_length",
    "train/completions/clipped_ratio",
]
OUTPUT_PATH = PROJECT_ROOT / "reward_model_training_metrics.csv"
KEEP_STEP = True
PAGE_SIZE = 1800 # how many rows to fetch at a time
DROP_ROWS_WITH_ALL_NAN_METRICS = True


def _iter_run_history_rows(
    run: wandb.apis.public.Run,
    page_size: int,
) -> Iterable[dict]:
    """Yield all run history rows from W&B Public API."""
    try:
        # Fetch full history rows first, then filter locally.
        yield from run.scan_history(page_size=page_size)
    except TypeError:
        # Fallback for older SDK signatures.
        yield from run.scan_history(page_size=page_size)


def fetch_and_filter_run_history(
    run_path: str,
    metric_names: list[str],
    output_path: str | Path,
    keep_step: bool = True,
    page_size: int = 1000,
    drop_rows_with_all_nan_metrics: bool = True,
) -> pd.DataFrame:
    """Fetch full W&B run history, keep selected metrics, and save to CSV."""
    if not metric_names:
        raise ValueError("metric_names must not be empty.")

    api = wandb.Api()
    run = api.run(run_path)
    history_key_info = run.history_keys.get("keys", {})
    available_columns = set(history_key_info.keys())

    requested_metric_columns = list(metric_names)
    matched_metric_columns = [
        column_name for column_name in requested_metric_columns if column_name in available_columns
    ]
    missing_metric_columns = [
        column_name for column_name in requested_metric_columns if column_name not in available_columns
    ]

    if missing_metric_columns:
        print("Warning: these metrics were not found in run history:")
        for column_name in missing_metric_columns:
            print(f"  - {column_name}")

    if not matched_metric_columns:
        available_train_columns = sorted(
            column_name
            for column_name in available_columns
            if isinstance(column_name, str) and column_name.startswith("train/")
        )
        raise ValueError(
            "None of METRIC_NAMES exists in this run. "
            f"Available train metrics: {available_train_columns}"
        )

    query_columns = list(matched_metric_columns)
    selected_columns = list(requested_metric_columns)
    if keep_step:
        query_columns = ["_step", *query_columns]
        selected_columns = ["_step", *selected_columns]

    rows = list(run.scan_history(keys=query_columns, page_size=page_size))
    history_df = pd.DataFrame(rows)

    # Ensure all requested output columns exist so schema stays stable.
    for column_name in selected_columns:
        if column_name not in history_df.columns:
            history_df[column_name] = pd.NA

    filtered_df = history_df[selected_columns].copy()

    if drop_rows_with_all_nan_metrics:
        if matched_metric_columns:
            filtered_df = filtered_df.dropna(subset=matched_metric_columns, how="all")
    if keep_step and "_step" in filtered_df.columns:
        filtered_df = filtered_df.drop_duplicates(subset=["_step"], keep="last")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    return filtered_df


def main() -> None:
    """Script entry point with in-file configuration."""
    filtered_df = fetch_and_filter_run_history(
        run_path=RUN_PATH,
        metric_names=METRIC_NAMES,
        output_path=OUTPUT_PATH,
        keep_step=KEEP_STEP,
        page_size=PAGE_SIZE,
        drop_rows_with_all_nan_metrics=DROP_ROWS_WITH_ALL_NAN_METRICS,
    )

    print(f"Saved filtered metrics to: {OUTPUT_PATH}")
    print(f"Rows: {len(filtered_df)}")
    print(f"Columns: {list(filtered_df.columns)}")


if __name__ == "__main__":
    main()
