# 06
import json
import argparse
from pathlib import Path
import logging
import pandas as pd
import seaborn as sns

# Force Matplotlib to use a non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_stats_data(processed_dir: Path) -> dict:
    """
    Finds all relevant stats files and aggregates all scores into master lists.
    """
    stats_files = [f for f in processed_dir.rglob("*_stats.json") if "parse_stats" not in f.name]
    logging.info(f"Found and processing {len(stats_files)} pipeline stats files.")

    if not stats_files:
        raise FileNotFoundError(f"No relevant stats files found in '{processed_dir}'. Please run the generation script first.")

    all_scores = {'similarity': [], 'faithfulness': [], 'correctness': [], 'judge': []}
    for file_path in stats_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
            all_scores['similarity'].extend(stats.get('deduplication_similarity_scores', []))
            all_scores['faithfulness'].extend(stats.get('ragas_faithfulness_scores', []))
            all_scores['correctness'].extend(stats.get('ragas_correctness_scores', []))
            all_scores['judge'].extend(stats.get('judge_scores', []))

    logging.info(
        f"Aggregated {len(all_scores['similarity'])} similarity scores and {len(all_scores['judge'])} judge scores.")
    return all_scores


def create_plots(scores: dict, output_path: Path):
    """
    Generates and saves a compact 2x2 subplot visualization with larger fonts.
    """
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.6)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # Increased height slightly for better spacing

    # Plot 1: Deduplication Similarity
    sns.histplot(ax=axes[0, 0], data=scores['similarity'], kde=True, bins=30)
    axes[0, 0].set_title('Distribution of Question Similarities')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Count')

    # Plot 2: RAGAS Faithfulness
    sns.histplot(ax=axes[0, 1], data=scores['faithfulness'], kde=True, bins=20, color='g')
    axes[0, 1].set_title('Distribution of RAGAS Faithfulness Scores')
    axes[0, 1].set_xlabel('Faithfulness Score')
    axes[0, 1].set_xlim(0, 1)

    # Plot 3: RAGAS Correctness
    sns.histplot(ax=axes[1, 0], data=scores['correctness'], kde=True, bins=20, color='r')
    axes[1, 0].set_title('Distribution of RAGAS Correctness Scores')
    axes[1, 0].set_xlabel('Correctness Score')
    axes[1, 0].set_xlim(0, 1)

    # Plot 4: LLM Judge Scores
    ax_judge = axes[1, 1]
    # Ensure all scores are integers for countplot
    judge_scores_int = [s for s in scores['judge'] if isinstance(s, int) and s >= 1]
    sns.countplot(ax=ax_judge, x=judge_scores_int, palette='viridis', order=[1, 2, 3, 4, 5])
    ax_judge.set_title('Distribution of LLM Judge Scores (Log Scale)')
    ax_judge.set_xlabel('Judge Score (1-5)')
    ax_judge.set_ylabel('Count (Log Scale)')
    ax_judge.set_yscale('log')
    # Add labels to the bars
    for p in ax_judge.patches:
        height = p.get_height()
        if height > 0: # Only annotate bars with a count > 0
            ax_judge.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                              ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                              textcoords='offset points')

    plt.tight_layout(pad=2.0) # Add padding between plots
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Successfully saved compact visualization to {output_path}")


def main():
    """Main function to orchestrate the visualization generation."""
    parser = argparse.ArgumentParser(description="Generate visualizations of quality scores from stats files.")
    parser.add_argument("--processed-dir", default="local_data/processed",
                        help="Directory containing processed document subfolders with stats files.")
    parser.add_argument("--output-file", default="paper_artifacts/figures/score_distributions.png",
                        help="Path to save the final visualization image.")
    args = parser.parse_args()

    try:
        all_scores = load_stats_data(Path(args.processed_dir))
        create_plots(all_scores, Path(args.output_file))
    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()