# 04
import json
import argparse
from pathlib import Path
import logging
import pandas as pd
import seaborn as sns
from scipy import stats

# Force Matplotlib to use a non-GUI backend
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_stats_data(processed_dir: Path) -> pd.DataFrame:
    """
    Finds all relevant stats files and extracts the page count and token count for each.
    """
    stats_files = [f for f in processed_dir.rglob("*_stats.json") if "parse_stats" not in f.name]
    logging.info(f"Found and processing {len(stats_files)} pipeline stats files.")

    if not stats_files:
        raise FileNotFoundError("No relevant stats files found. Please run the main generation script first.")

    all_doc_data = []
    for file_path in stats_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if 'manual_id' in data and 'input_pages' in data and 'prompt_token_count' in data:
                    all_doc_data.append({
                        "manual_id": data['manual_id'],
                        "num_pages": data['input_pages'],
                        "token_count": data['prompt_token_count']
                    })
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Skipping stats file {file_path} due to error: {e}")

    return pd.DataFrame(all_doc_data)


def estimate_max_pages(df: pd.DataFrame):
    """
    Performs a linear regression and estimates the number of pages to reach 1M tokens.
    """
    if df.empty or len(df) < 2:
        logging.warning("Not enough data points to perform linear regression.")
        return

    slope, intercept, r_value, p_value, std_err = stats.linregress(df['num_pages'], df['token_count'])
    r_squared = r_value ** 2

    if slope > 0:
        max_pages = (1_000_000 - intercept) / slope
        print("\n--- Pipeline Scalability Estimation ---")
        print(f"Linear regression model fit (R-squared): {r_squared:.4f}")
        print(
            f"Based on the trend, the estimated maximum manual size for a 1M token context is: ~{int(max_pages)} pages.")
        print("---------------------------------------\n")
    else:
        logging.warning("Regression slope is not positive; cannot estimate maximum page count.")


def create_plot(df: pd.DataFrame, output_path: Path):
    """
    Generates and saves a compact scatter plot with larger, paper-quality fonts.
    """
    if df.empty:
        logging.error("Cannot create plot, no data was loaded.")
        return

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.4)

    plt.figure(figsize=(8, 4))

    plot = sns.regplot(
        data=df, x='num_pages', y='token_count', ci=95,
        scatter_kws={'s': 100, 'alpha': 0.8, 'edgecolor': 'w'}
    )

    # --- REMOVED: Loop for annotating individual points ---

    plt.title('Prompt Token Count as a Function of Document Length')
    plt.xlabel('Number of Pages in Source Manual')
    plt.ylabel('Total Tokens in Generator Prompt')
    plt.ylim(0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Successfully saved clean visualization to {output_path}")


def main():
    """Main function to orchestrate the visualization generation."""
    parser = argparse.ArgumentParser(description="Generate a plot of prompt tokens vs. page count.")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directory containing processed document subfolders.")
    parser.add_argument("--output-file", default="paper_artifacts/figures/prompt_size_vs_pages.png",
                        help="Path to save the final visualization image.")
    args = parser.parse_args()

    try:
        stats_df = load_stats_data(Path(args.processed_dir))
        if not stats_df.empty:
            estimate_max_pages(stats_df)
            create_plot(stats_df, Path(args.output_file))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()