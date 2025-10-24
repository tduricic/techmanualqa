"""
Ablation analysis script to analyze filter complementarity.

This script analyzes the ablation logs to determine how many candidates
were rejected by the LLM Judge only, RAGAS only, or both filters.
"""

import pandas as pd
from pathlib import Path
import sys


def main():
    """Main entry point for the ablation analysis script."""

    # --- Configuration ---
    # These values MUST match the thresholds in your main pipeline script
    RAGAS_FAIL_THRESHOLD = 0.80
    JUDGE_FAIL_THRESHOLD = 4  # Fails if score is < 4
    JUDGE_PASS_THRESHOLD = 4  # Passes if score is >= 4

    # This is the category that was SKIPPED by the filters
    UNANSWERABLE_CATEGORY_NAME = "Unanswerable"

    # --- Script ---
    DATA_DIR = Path("data/processed")
    # This pattern now searches all subdirectories
    FILE_PATTERN = "**/*_ablation_log.jsonl"

    print(f"--- Ablation Proxy Analysis ---")
    print(f"Recursively searching for log files in: {DATA_DIR.resolve()}")

    all_dfs = []
    log_files = list(DATA_DIR.glob(FILE_PATTERN))

    if not log_files:
        print(f"\nError: No log files found matching '{FILE_PATTERN}' in {DATA_DIR}")
        print("Please check the DATA_DIR path in this script.")
        sys.exit(1)

    print(f"Found {len(log_files)} ablation log files. Loading...")

    for f in log_files:
        try:
            df = pd.read_json(f, lines=True)
            all_dfs.append(df)
            print(f"  Loaded {f.name}")
        except Exception as e:
            print(f"  Warning: Could not read {f}: {e}")

    if not all_dfs:
        print("Error: No data was loaded. Exiting.")
        sys.exit(1)

    # Combine all data into one massive DataFrame
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total candidates loaded from all manuals: {len(full_df)}")

    # --- Analysis ---

    # 1. Filter to *only* the 'Answerable' candidates that were evaluated
    if 'category' not in full_df.columns:
        print("Error: 'category' column not found. Cannot proceed.")
        sys.exit(1)

    eval_df = full_df[full_df['category'] != UNANSWERABLE_CATEGORY_NAME].copy()
    total_evaluated = len(eval_df)

    if total_evaluated == 0:
        print("Error: No 'Answerable' candidates found to analyze.")
        sys.exit(1)

    print(f"\nAnalyzing {total_evaluated} 'Answerable' candidates...")

    # 2. Define failure and pass conditions
    #    (Using >= 0 to filter out any potential -1s from un-scored rows)
    eval_df['failed_ragas'] = (
                                      (eval_df['ragas_faithfulness'] < RAGAS_FAIL_THRESHOLD) |
                                      (eval_df['ragas_correctness'] < RAGAS_FAIL_THRESHOLD)
                              ) & (eval_df['ragas_faithfulness'] >= 0)

    eval_df['failed_judge'] = (
                                      eval_df['judge_score'] < JUDGE_FAIL_THRESHOLD
                              ) & (eval_df['judge_score'] >= 0)

    eval_df['passed_ragas'] = ~eval_df['failed_ragas']
    eval_df['passed_judge'] = ~eval_df['failed_judge']

    # 3. Calculate the sets
    judge_only_rejects = eval_df[eval_df['failed_judge'] & eval_df['passed_ragas']]
    ragas_only_rejects = eval_df[eval_df['failed_ragas'] & eval_df['passed_judge']]
    both_rejects = eval_df[eval_df['failed_ragas'] & eval_df['failed_judge']]

    count_x = len(judge_only_rejects)
    count_y = len(ragas_only_rejects)
    count_overlap = len(both_rejects)

    # 4. Calculate denominators for the paper's percentages
    total_rejected_by_judge = count_x + count_overlap
    total_rejected_by_ragas = count_y + count_overlap
    total_rejected_overall = count_x + count_y + count_overlap

    # 5. Handle potential division by zero
    percent_x = (count_x / total_rejected_by_judge * 100) if total_rejected_by_judge > 0 else 0
    percent_y = (count_y / total_rejected_by_ragas * 100) if total_rejected_by_ragas > 0 else 0
    percent_overlap_of_judge = (count_overlap / total_rejected_by_judge * 100) if total_rejected_by_judge > 0 else 0
    percent_overlap_of_ragas = (count_overlap / total_rejected_by_ragas * 100) if total_rejected_by_ragas > 0 else 0

    # --- Print Report ---
    print("\n" + "=" * 40)
    print("--- Filter Complementarity Report ---")
    print("=" * 40)
    print(f"Total 'Answerable' Candidates Analyzed: {total_evaluated}\n")

    print(f"Total Candidates Rejected by *at least one* filter: {total_rejected_overall}")
    print(f"  - Rejected by LLM Judge ONLY (Passed RAGAS): {count_x}")
    print(f"  - Rejected by RAGAS ONLY (Passed LLM Judge): {count_y}")
    print(f"  - Rejected by BOTH (Overlap):                {count_overlap}")

    print("\n" + "=" * 40)
    print("--- STATISTICS FOR YOUR PAPER ---")
    print("=" * 40)
    print("Copy and paste these numbers into your `Ablation Proxy` paragraph:\n")

    print(f"Total candidates rejected by the LLM Judge (Score < {JUDGE_FAIL_THRESHOLD}): {total_rejected_by_judge}")
    print(f"Total candidates rejected by RAGAS (Score < {RAGAS_FAIL_THRESHOLD}):         {total_rejected_by_ragas}")
    print("\n------------------------------------------------------------------")
    print(
        f"\"We found that {percent_x:.1f}% of the candidates rejected by the LLM Judge ({count_x}/{total_rejected_by_judge}) had actually passed the RAGAS threshold.")
    print(
        f"Conversely, {percent_y:.1f}% of the candidates rejected by RAGAS ({count_y}/{total_rejected_by_ragas}) had received a high score (>= {JUDGE_PASS_THRESHOLD}) from the LLM Judge.\"")
    print("------------------------------------------------------------------\n")

    print("\n--- Additional Context (Venn Diagram numbers) ---")
    print(f"  - LLM Judge Rejects that RAGAS *missed*: {count_x} ({percent_x:.1f}%)")
    print(f"  - RAGAS Rejects that LLM Judge *missed*: {count_y} ({percent_y:.1f}%)")
    print(f"  - Overlap (Rejected by Both):            {count_overlap}")
    print(f"  - % of Judge rejects that were overlap:  {percent_overlap_of_judge:.1f}%")
    print(f"  - % of RAGAS rejects that were overlap:  {percent_overlap_of_ragas:.1f}%")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()