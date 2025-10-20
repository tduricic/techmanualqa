# 08
import json
import argparse
from pathlib import Path
import logging
import pandas as pd
import shutil

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def write_readme(output_dir: Path):
    """Creates the README.md file with dataset description."""
    readme_content = """
# TechManualQA-350 Dataset

This archive contains the data artifacts for the CIKM 2025 paper: "A Semi-Automated Pipeline for Generating Grounded, Structurally-Aware QA Datasets for Technical Manuals".

## Contents

- `TechManualQA_350.jsonl`: The complete benchmark dataset containing 350 question-answering pairs from 10 technical manuals. Each line is a JSON object.
- `human_annotation/`: A directory containing the raw, filled-out Excel files from our two annotators (A and B) for both the general and procedural audit tasks. This data is provided for full reproducibility of our inter-rater reliability analysis.
- `LICENSE`: The license for this dataset (CC BY 4.0).

## Data Schema for `TechManualQA_350.jsonl`

Each JSON object in the main dataset file contains the following keys:
- `question_id`: A unique identifier for the question.
- `doc_name`: The name of the source manual.
- `question_text`: The generated question.
- `gt_answer_snippet`: The verbatim answer snippet extracted from the manual.
- `gt_page_number`: The page number in the source manual where the answer was found.
- `category`: The question category (e.g., "Procedural Step Inquiry").
- `persona`: The persona used for generation (e.g., "Technician").
- `parsed_steps`: A list of strings representing the parsed steps for procedural questions (null otherwise).
- `annotator_A_*`, `annotator_B_*`: For the subset of questions included in our human audit, these fields contain the labels provided by each annotator (e.g., `annotator_A_correctness`). These fields are `null` for non-audited questions.
"""
    (output_dir / "README.md").write_text(readme_content.strip(), encoding='utf-8')
    logging.info(f"Created README.md in {output_dir}")


def write_license(output_dir: Path):
    """Creates the LICENSE file."""
    license_content = """
Creative Commons Attribution 4.0 International (CC BY 4.0)

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material
for any purpose, even commercially.

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

For the full license text, see: https://creativecommons.org/licenses/by/4.0/
"""
    (output_dir / "LICENSE").write_text(license_content.strip(), encoding='utf-8')
    logging.info(f"Created LICENSE in {output_dir}")


def main():
    """
    Main function to find all necessary files, create the final dataset,
    and package everything into a Zenodo-ready zip file.
    """
    parser = argparse.ArgumentParser(description="Create a Zenodo-ready archive for the TechManualQA dataset.")
    parser.add_argument("--processed-dir", default="local_data/processed",
                        help="Directory containing the candidate JSONL files.")
    parser.add_argument("--audit-dir", default="human_validation/audit_sheets",
                        help="Directory containing the filled-out audit Excel files.")
    parser.add_argument("--output-name", default="TechManualQA_350",
                        help="The base name for the output directory and zip file.")
    args = parser.parse_args()

    # --- Define paths (with recommended output directory) ---
    processed_dir = Path(args.processed_dir)
    audit_dir = Path(args.audit_dir)
    release_dir = Path("release")  # All final outputs will go here
    output_dir = release_dir / args.output_name  # e.g., release/TechManualQA_350/
    zip_path_base = release_dir / args.output_name  # e.g., release/TechManualQA_350 (for zip)

    # --- Step 1: Create the output directory structure ---
    if output_dir.exists():
        logging.warning(f"Output directory '{output_dir}' already exists. It will be overwritten.")
        shutil.rmtree(output_dir)

    release_dir.mkdir(exist_ok=True)
    human_annotation_path = output_dir / "human_annotation"
    human_annotation_path.mkdir(parents=True)
    logging.info(f"Created directory structure at '{output_dir}'")

    # --- Step 2: Copy the raw audit files ---
    try:
        if not any(audit_dir.glob("*.xlsx")):
            raise FileNotFoundError(f"No .xlsx files found in '{audit_dir}'")
        for f in audit_dir.glob("*.xlsx"):
            shutil.copy(f, human_annotation_path)
        logging.info(f"Copied human annotation files from '{audit_dir}'")
    except FileNotFoundError as e:
        logging.error(f"Audit files not found. Please run the audit preparation script first. Details: {e}")
        return

    # --- Step 3: Load all candidate and annotation data ---
    all_candidates = {}
    candidate_files = list(processed_dir.rglob("*_candidates.jsonl"))
    if not candidate_files:
        logging.error(f"No candidate files found in '{processed_dir}'. Please run the main pipeline first.")
        return

    for file_path in candidate_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line)
                all_candidates[task['question_id']] = task

    df_gen_a = pd.read_excel(audit_dir / "general_audit_A.xlsx").set_index("question_id")
    df_gen_b = pd.read_excel(audit_dir / "general_audit_B.xlsx").set_index("question_id")
    df_proc_a = pd.read_excel(audit_dir / "procedural_audit_A.xlsx").set_index("question_id")
    df_proc_b = pd.read_excel(audit_dir / "procedural_audit_B.xlsx").set_index("question_id")
    logging.info(f"Loaded {len(all_candidates)} candidate questions and all annotation data.")

    # --- Step 4: Create the final adjudicated dataset ---
    final_dataset = []
    for qid in sorted(all_candidates.keys()):
        task = all_candidates[qid]
        # Set default null values for all possible annotation fields
        for annotator in ["A", "B"]:
            task[f'annotator_{annotator}_correctness'] = None
            task[f'annotator_{annotator}_groundedness'] = None
            task[f'annotator_{annotator}_quality'] = None
            task[f'annotator_{annotator}_snippet_correctness'] = None
            task[f'annotator_{annotator}_parsing_quality'] = None

        if qid in df_gen_a.index:
            task['annotator_A_correctness'] = df_gen_a.loc[qid, 'Answer Correctness']
            task['annotator_A_groundedness'] = df_gen_a.loc[qid, 'Answer Groundedness']
            task['annotator_A_quality'] = df_gen_a.loc[qid, 'Question Quality']
            task['annotator_B_correctness'] = df_gen_b.loc[qid, 'Answer Correctness']
            task['annotator_B_groundedness'] = df_gen_b.loc[qid, 'Answer Groundedness']
            task['annotator_B_quality'] = df_gen_b.loc[qid, 'Question Quality']
        elif qid in df_proc_a.index:
            task['annotator_A_snippet_correctness'] = df_proc_a.loc[qid, 'Answer Snippet Correctness']
            task['annotator_A_parsing_quality'] = df_proc_a.loc[qid, 'Step Parsing Quality']
            task['annotator_A_quality'] = df_proc_a.loc[qid, 'Question Quality']
            task['annotator_B_snippet_correctness'] = df_proc_b.loc[qid, 'Answer Snippet Correctness']
            task['annotator_B_parsing_quality'] = df_proc_b.loc[qid, 'Step Parsing Quality']
            task['annotator_B_quality'] = df_proc_b.loc[qid, 'Question Quality']

        final_dataset.append(task)

    # Write the final JSONL file
    (output_dir / f"{args.output_name}.jsonl").write_text(
        "\n".join(json.dumps(entry, default=str) for entry in final_dataset),
        encoding='utf-8'
    )
    logging.info(f"Created final dataset file with {len(final_dataset)} entries.")

    # --- Step 5: Write documentation and license ---
    write_readme(output_dir)
    write_license(output_dir)

    # --- Step 6: Create the final ZIP archive ---
    shutil.make_archive(str(zip_path_base), 'zip', root_dir=release_dir, base_dir=args.output_name)
    logging.info(f"Successfully created final archive: {zip_path_base.with_suffix('.zip')}")
    logging.info("You can now find the Zenodo-ready archive in the 'release/' directory.")


if __name__ == "__main__":
    main()