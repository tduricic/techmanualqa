import json
import argparse
from pathlib import Path
import logging
import pandas as pd

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_latex_table(average_stats: pd.Series, output_path: Path):
    """
    Generates and saves a publication-ready, resizable LaTeX table from the average stats.
    """
    stage_names = {
        'rows_parsed': 'Initial Parsed Generation',
        'rows_after_meta_filter': 'After Metadata Filter',
        'rows_after_dedupe': 'After Semantic Deduplication',
        'rows_after_ragas': 'After RAGAS Grounding Filter',
        'rows_after_judge': 'After LLM Judge Filter',
        'rows_after_unanswerable_verify': 'After Unanswerable Verification',
        'final_dataset_size_generated': 'After Quota Selection (Final Set)' # Refined name
    }

    # Reorder the keys to match the pipeline flow for the table
    ordered_keys = [
        'rows_parsed', 'rows_after_meta_filter', 'rows_after_dedupe',
        'rows_after_ragas', 'rows_after_judge', 'rows_after_unanswerable_verify',
        'final_dataset_size_generated'
    ]

    latex_lines = [
        "\\begin{table}[t!]", # Use [t!] for top-of-page placement
        "\\centering",
        "\\caption{Average number of candidate QA pairs remaining per manual after key filtering stages.}",
        "\\label{tab:filter_stats}",
        "\\resizebox{0.7\\columnwidth}{!}{%", # Use a fraction of columnwidth for better control
        "\\begin{tabular}{lr}",
        "\\toprule",
        "\\textbf{Pipeline Stage} & \\textbf{Avg. QA Pairs} \\\\",
        "\\midrule"
    ]

    for key in ordered_keys:
        if key in average_stats and key in stage_names:
            stage_name = stage_names[key]
            avg_value = f"{average_stats[key]:.1f}"
            # Using ljust for alignment is fine, but LaTeX handles spacing.
            # Keeping it simple is often better.
            latex_lines.append(f"{stage_name} & {avg_value} \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}"
    ])

    final_latex = "\n".join(latex_lines)

    output_path.write_text(final_latex, encoding='utf-8')

    logging.info(f"LaTeX table saved to {output_path}")
    print("\n--- Generated LaTeX Table Code ---\n")
    print(final_latex)


def main():
    """
    Main function to find stats files, calculate averages, and generate the LaTeX table.
    """
    parser = argparse.ArgumentParser(description="Process stats files and generate a filtering summary table.")
    parser.add_argument("--processed-dir", default="local_data/processed",
                        help="Directory containing processed document subfolders.")
    parser.add_argument("--output-file", default="paper_artifacts/tables/filtering_stats_table.tex",
                        help="Path to save the final .tex file.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_tex_path = Path(args.output_file)

    # NEW: Ensure the output directory exists
    output_tex_path.parent.mkdir(parents=True, exist_ok=True)

    all_found_files = list(processed_dir.rglob("*_stats.json"))
    stats_files = [f for f in all_found_files if "parse_stats" not in f.name]

    if not stats_files:
        logging.error(f"No pipeline stats files found in '{processed_dir}'. Please run the generation script first.")
        return

    logging.info(f"Found {len(stats_files)} relevant pipeline stats files to analyze.")

    keys_to_extract = [
        'rows_parsed', 'rows_after_meta_filter', 'rows_after_dedupe', 'rows_after_ragas',
        'rows_after_judge', 'rows_after_unanswerable_verify', 'final_dataset_size_generated'
    ]

    all_stats_data = []
    for file_path in stats_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                # Ensure the file is from a completed run to avoid partial data
                if stats.get('pipeline_status') == 'Generated' or stats.get('quota_met_fully') is True:
                    doc_stats = {key: stats.get(key, 0) for key in keys_to_extract}
                    all_stats_data.append(doc_stats)
                else:
                    logging.warning(f"Skipping stats file (not complete or failed): {file_path.name}")
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Could not process file {file_path.name}, skipping. Reason: {e}")

    if not all_stats_data:
        logging.error("No valid stats data could be extracted from completed runs.")
        return

    logging.info(f"Successfully processed {len(all_stats_data)} valid stats files.")

    df_stats = pd.DataFrame(all_stats_data)
    average_stats = df_stats.mean()

    # --- ENHANCEMENT: Save the raw summary data to a JSON file ---
    # Derive the JSON filename from the TeX output filename
    output_json_path = output_tex_path.with_suffix('.json')
    try:
        average_stats.to_json(output_json_path, indent=2)
        logging.info(f"Machine-readable summary data saved to: {output_json_path}")
    except Exception as e:
        logging.error(f"Failed to save summary JSON file: {e}")
    # --- End of Enhancement ---

    # Create the LaTeX table using the calculated averages
    create_latex_table(average_stats, output_tex_path)


if __name__ == "__main__":
    main()