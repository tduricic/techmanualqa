import pandas as pd
from sklearn.metrics import cohen_kappa_score
import argparse
from pathlib import Path
import logging
import json  # Added json import

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_dist_and_kappa(df_a: pd.DataFrame, df_b: pd.DataFrame, column_name: str, choices_map: list) -> (list, float):
    """
    Calculates distribution on agreed-upon data and the Cohen's Kappa score.
    This version is robust to sorting and data type issues.
    """
    if column_name not in df_a.columns or column_name not in df_b.columns:
        logging.warning(f"Column '{column_name}' not found. Skipping.")
        return None, 0.0

    # Align dataframes on their index. This is crucial if rows are not in the same order.
    aligned_a, aligned_b = df_a.align(df_b, join='inner', axis=0)

    # Create a mask to only include rows where both annotators provided a label for the specific column
    mask = aligned_a[column_name].notna() & aligned_b[column_name].notna()

    if mask.sum() == 0:
        logging.warning(f"No common annotated data for '{column_name}'.")
        return None, 0.0

    # Calculate Kappa score on the aligned, non-null data
    kappa = cohen_kappa_score(aligned_a.loc[mask, column_name], aligned_b.loc[mask, column_name])

    # Create a gold standard dataset based on rows where annotators agreed
    agreement_mask = (aligned_a.loc[mask, column_name] == aligned_b.loc[mask, column_name])
    agreed_indices = aligned_a.loc[mask][agreement_mask].index
    df_gold = df_a.loc[agreed_indices].copy()

    if df_gold.empty:
        logging.warning(f"No agreements found for column '{column_name}'.")
        return [0.0] * len(choices_map), kappa  # Return zero distribution but valid kappa

    # Calculate the percentage distribution of choices
    dist = df_gold[column_name].value_counts(normalize=True).mul(100)
    dist_ordered = [dist.get(choice, 0) for choice in choices_map]

    return dist_ordered, kappa


def create_latex_table(general_results: dict, procedural_results: dict, output_path: Path):
    """Generates and saves a publication-ready LaTeX table."""
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latex_lines = [
        "\\begin{table}[!t]", "\\centering", "\\caption{",
        "  Human evaluation of QA generation and parsing quality. Agreement is reported as Cohen's Kappa ($\\kappa$).",
        "  Results are percentages of the adjudicated dataset where annotators agreed.", "}",
        "\\label{tab:human_eval_results}", "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
        "\\textbf{Metric} ($\\kappa$) & \\textbf{Success (\\%)} & \\textbf{Partial (\\%)} & \\textbf{Failure (\\%)} \\\\ \\midrule",
        "\\multicolumn{4}{l}{\\textit{General Audit}} \\\\",
    ]
    # General Audit Results
    gen_correctness = general_results.get("Answer Correctness", {})
    gen_grounded = general_results.get("Answer Groundedness", {})
    gen_quality = general_results.get("Question Quality", {})

    if gen_correctness:
        line = f"  Answer Correctness ({gen_correctness.get('kappa', 0):.2f}) & {gen_correctness.get('dist', [0, 0, 0])[0]:.1f} & {gen_correctness.get('dist', [0, 0, 0])[1]:.1f} & {gen_correctness.get('dist', [0, 0, 0])[2]:.1f} \\\\"
        latex_lines.append(line)
    if gen_grounded:
        line = f"  Answer Groundedness ({gen_grounded.get('kappa', 0):.2f}) & {gen_grounded.get('dist', [0, 0, 0])[0]:.1f} & {gen_grounded.get('dist', [0, 0, 0])[1]:.1f} & {gen_grounded.get('dist', [0, 0, 0])[2]:.1f} \\\\"
        latex_lines.append(line)
    if gen_quality:
        line = f"  Question Quality ({gen_quality.get('kappa', 0):.2f}) & {gen_quality.get('dist', [0, 0])[0]:.1f} & -- & {gen_quality.get('dist', [0, 0])[1]:.1f} \\\\"
        latex_lines.append(line)

    latex_lines.append("\\midrule")
    latex_lines.append("\\multicolumn{4}{l}{\\textit{Procedural Audit}} \\\\")

    # Procedural Audit Results
    proc_snippet = procedural_results.get("Answer Snippet Correctness", {})
    proc_parsing = procedural_results.get("Step Parsing Quality", {})
    proc_quality = procedural_results.get("Question Quality", {})

    if proc_snippet:
        line = f"  Answer Snippet Correctness ({proc_snippet.get('kappa', 0):.2f}) & {proc_snippet.get('dist', [0, 0, 0])[0]:.1f} & {proc_snippet.get('dist', [0, 0, 0])[1]:.1f} & {proc_snippet.get('dist', [0, 0, 0])[2]:.1f} \\\\"
        latex_lines.append(line)
    if proc_parsing:
        line = f"  Step Parsing Quality ({proc_parsing.get('kappa', 0):.2f}) & {proc_parsing.get('dist', [0, 0, 0])[0]:.1f} & {proc_parsing.get('dist', [0, 0, 0])[1]:.1f} & {proc_parsing.get('dist', [0, 0, 0])[2]:.1f} \\\\"
        latex_lines.append(line)
    if proc_quality:
        line = f"  Question Quality ({proc_quality.get('kappa', 0):.2f}) & {proc_quality.get('dist', [0, 0])[0]:.1f} & -- & {proc_quality.get('dist', [0, 0])[1]:.1f} \\\\"
        latex_lines.append(line)

    latex_lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    final_latex = "\n".join(latex_lines)

    output_path.write_text(final_latex, encoding='utf-8')
    logging.info(f"LaTeX table saved to {output_path}")
    print("\n--- Generated LaTeX Table Code ---\n")
    print(final_latex)


def main():
    parser = argparse.ArgumentParser(description="Process annotated audit files and generate a LaTeX results table.")
    parser.add_argument("--audit-dir", default="human_validation/audit_sheets",
                        help="Directory containing the filled-out audit Excel files.")
    parser.add_argument("--output-file", default="paper_artifacts/tables/human_audit_results.tex",
                        help="Path to save the final .tex file.")
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    output_tex_path = Path(args.output_file)

    try:
        # Load data, ignoring sentinels by filtering for integer question_id
        df_gen_a = pd.read_excel(audit_dir / "general_audit_A.xlsx").set_index('question_id')
        df_gen_b = pd.read_excel(audit_dir / "general_audit_B.xlsx").set_index('question_id')
        df_proc_a = pd.read_excel(audit_dir / "procedural_audit_A.xlsx").set_index('question_id')
        df_proc_b = pd.read_excel(audit_dir / "procedural_audit_B.xlsx").set_index('question_id')
    except FileNotFoundError as e:
        logging.error(
            f"Error loading files from '{audit_dir}'. Make sure all 4 audit Excel files are present. Details: {e}")
        return

    # Define the structure for validation
    general_questions = {
        "Answer Correctness": ["Correct", "Partially Correct", "Incorrect"],
        "Answer Groundedness": ["Fully Grounded", "Partially Grounded (contains outside info)", "Not Grounded"],
        "Question Quality": ["Clear & Usable", "Unclear or Flawed"]
    }
    procedural_questions = {
        "Answer Snippet Correctness": ["Correct", "Partially Correct", "Incorrect"],
        "Step Parsing Quality": ["Correctly Parsed", "Partially Correct", "Incorrectly Parsed"],
        "Question Quality": ["Clear & Usable", "Unclear or Flawed"]
    }

    # Data Cleaning: Strip whitespace from all relevant annotation columns
    all_q_columns = list(general_questions.keys()) + list(procedural_questions.keys())
    for df in [df_gen_a, df_gen_b, df_proc_a, df_proc_b]:
        for col in all_q_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.strip()

    # Calculate results
    general_results, procedural_results = {}, {}
    for question, choices in general_questions.items():
        dist, kappa = get_dist_and_kappa(df_gen_a, df_gen_b, question, choices)
        if dist is not None: general_results[question] = {'dist': dist, 'kappa': kappa}
    for question, choices in procedural_questions.items():
        dist, kappa = get_dist_and_kappa(df_proc_a, df_proc_b, question, choices)
        if dist is not None: procedural_results[question] = {'dist': dist, 'kappa': kappa}

    # --- ENHANCEMENT: Save summary data to JSON ---
    summary_data = {
        "general_audit_results": general_results,
        "procedural_audit_results": procedural_results
    }
    output_json_path = output_tex_path.with_suffix('.json')
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        logging.info(f"Machine-readable summary of audit results saved to: {output_json_path}")
    except Exception as e:
        logging.error(f"Failed to save summary JSON file: {e}")

    # Create the LaTeX table
    create_latex_table(general_results, procedural_results, output_tex_path)


if __name__ == "__main__":
    main()