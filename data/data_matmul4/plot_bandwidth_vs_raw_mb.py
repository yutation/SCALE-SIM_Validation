#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_merge_copy_events() -> pd.DataFrame:
    """
    Load the three filtered copy event CSVs, tag them with dataset names,
    and return a single concatenated DataFrame.
    """
    base_dir = Path(__file__).parent

    csv_small = base_dir / "filtered_events_copy_done.csv"
    csv_medium = base_dir / "filtered_events_copy_done2.csv"
    csv_large = base_dir / "filtered_events_copy_done3.csv"

    df_small = pd.read_csv(csv_small)
    df_medium = pd.read_csv(csv_medium)
    df_large = pd.read_csv(csv_large)

    df_small["dataset"] = "small"
    df_medium["dataset"] = "medium"
    df_large["dataset"] = "large"

    combined_df = pd.concat([df_small, df_medium, df_large], ignore_index=True)
    return combined_df


def plot_bandwidth_vs_raw_mb(merged_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate a single scatter plot of Bandwidth (GB/s) vs Raw byte access (MB)
    and save to output_path.
    """
    df = merged_df.copy()

    # Convert raw bytes to megabytes (MB, decimal)
    df["raw_mb"] = df["raw_bytes_accessed"] / 1_000_000.0

    plt.style.use("default")
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(8, 6))
    for dataset_name in df["dataset"].unique():
        subset = df[df["dataset"] == dataset_name]
        ax.scatter(
            subset["raw_mb"],
            subset["bandwidth_gbps"],
            label=dataset_name.capitalize(),
            alpha=0.75,
            s=40,
        )

    ax.set_xlabel("Raw byte access (MB)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Bandwidth vs Raw byte access (MB)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Dataset")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def main() -> None:
    merged_df = load_and_merge_copy_events()
    output_png = Path(__file__).parent / "bandwidth_vs_raw_mb.png"
    plot_bandwidth_vs_raw_mb(merged_df, output_png)
    print(f"Saved figure to: {output_png}")


if __name__ == "__main__":
    main()


