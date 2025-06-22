import json
import argparse
from pathlib import Path
import logging
import pandas as pd
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Font, Alignment

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def write_instructions(worksheet):
    """Adds a formatted instruction sheet to the Excel workbook."""
    worksheet.title = "Instructions"

    # Set column widths for readability
    worksheet.column_dimensions['A'].width = 30
    worksheet.column_dimensions['B'].width = 80

    # Define content: {Cell: (Text, is_bold)}
    instructions = {
        'A1': ("General Guidelines", True),
        'A2': ("Task:", False),
        'B2': (
        "For each row, please review the Question, Generated Answer, and the source document context. Then, select the most appropriate option from the dropdown menu for each annotation column.",
        False),
        'A4': ("Annotation for General Questions", True),
        'A5': ("Answer Correctness", True),
        'B5': ("Correct: The answer is factually and completely correct based on the text.", False),
        'B6': ("Partially Correct: The answer is mostly correct but contains minor inaccuracies or omissions.", False),
        'B7': ("Incorrect: The answer is factually wrong or misleading.", False),
        'A8': ("Answer Groundedness", True),
        'B8': ("Fully Grounded: All information in the answer comes directly from the page text.", False),
        'B9': (
        "Partially Grounded: The answer is mostly from the page text but includes some outside information.", False),
        'B10': ("Not Grounded: The answer is not supported by the page text.", False),
        'A11': ("Question Quality", True),
        'B11': ("Clear & Usable: The question is well-formed, unambiguous, and makes sense.", False),
        'B12': ("Unclear or Flawed: The question is confusing, ambiguous, or grammatically incorrect.", False),
        'A14': ("Annotation for Procedural Questions", True),
        'A15': ("Answer Snippet Correctness", True),
        'B15': ("Same as 'Answer Correctness' above. Evaluates the original text snippet.", False),
        'A16': ("Step Parsing Quality", True),
        'B16': ("Correctly Parsed: The parsed steps perfectly match the snippet in number, content, and order.", False),
        'B17': ("Incorrectly Parsed: The parsed steps have errors in number, content, or order.", False),
    }

    for cell, (text, is_bold) in instructions.items():
        worksheet[cell].value = text
        worksheet[cell].font = Font(bold=is_bold, size=12)
        worksheet[cell].alignment = Alignment(wrap_text=True, vertical='top')


def create_xlsx_audit_file(tasks: list, columns_to_show: list, validation_map: dict, output_path: Path):
    """Creates a formatted Excel file with dropdown menus and an instruction sheet."""
    if not tasks:
        logging.warning(f"No tasks provided for {output_path.name}. Skipping file creation.")
        return

    df = pd.DataFrame(tasks)
    for col_name in validation_map.keys():
        df[col_name] = ""
    final_columns = columns_to_show + list(validation_map.keys())
    df = df.reindex(columns=final_columns)

    logging.info(f"Creating Excel file: {output_path} with {len(df)} rows.")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Annotation')
        workbook = writer.book
        worksheet = writer.sheets['Annotation']

        # --- Add Instructions Sheet ---
        instruction_sheet = workbook.create_sheet("Instructions")
        write_instructions(instruction_sheet)

        # --- Add Formatting and Data Validation (Dropdowns) ---
        worksheet.freeze_panes = 'A2'
        col_map = {col.value: col.column_letter for col in worksheet[1]}

        for col_name, choices in validation_map.items():
            if col_name not in col_map: continue
            col_letter = col_map[col_name]
            validation_formula = f'"{",".join(choices)}"'
            dv = DataValidation(type="list", formula1=validation_formula, allow_blank=True)
            worksheet.add_data_validation(dv)
            cell_range = f"{col_letter}2:{col_letter}{worksheet.max_row + 100}"  # Apply to more rows
            dv.add(cell_range)

        for column_cells in worksheet.columns:
            header_text = column_cells[0].value
            worksheet.column_dimensions[
                column_cells[0].column_letter].width = 25 if header_text in validation_map else 40
            column_cells[0].font = Font(bold=True)

    logging.info(f"Successfully saved and formatted {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare audit data in Excel format for annotation.")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directory containing processed document subfolders.")
    parser.add_argument("--output-dir", default="human_validation/audit_sheets",
                        help="Directory where final Excel audit files will be saved.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    GENERAL_CHOICES = {
        "Answer Correctness": ["Correct", "Partially Correct", "Incorrect"],
        "Answer Groundedness": ["Fully Grounded", "Partially Grounded (contains outside info)", "Not Grounded"],
        "Question Quality": ["Clear & Usable", "Unclear or Flawed"]
    }
    PROCEDURAL_CHOICES = {
        "Question Quality": ["Clear & Usable", "Unclear or Flawed"],
        "Answer Snippet Correctness": ["Correct", "Partially Correct", "Incorrect"],
        "Step Parsing Quality": ["Correctly Parsed", "Partially Correct","Incorrectly Parsed"]
    }

    general_tasks_by_doc, procedural_tasks_by_doc = {}, {}
    for file_path in processed_dir.rglob("*_general_audit_A.jsonl"):
        doc_name = file_path.stem.replace("_general_audit_A", "")
        if doc_name not in general_tasks_by_doc: general_tasks_by_doc[doc_name] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line)
                task['doc_name'] = doc_name
                general_tasks_by_doc[doc_name].append(task)

    for file_path in processed_dir.rglob("*_procedural_audit_A.jsonl"):
        doc_name = file_path.stem.replace("_procedural_audit_A", "")
        if doc_name not in procedural_tasks_by_doc: procedural_tasks_by_doc[doc_name] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line)
                task['doc_name'] = doc_name
                procedural_tasks_by_doc[doc_name].append(task)

    final_general_tasks, final_procedural_tasks = [], []
    for doc_name in sorted(general_tasks_by_doc.keys()):
        tasks = general_tasks_by_doc[doc_name]
        tasks.sort(key=lambda t: int(t.get('gt_page_number', -1)))
        final_general_tasks.extend(tasks)

    for doc_name in sorted(procedural_tasks_by_doc.keys()):
        tasks = procedural_tasks_by_doc[doc_name]
        tasks.sort(key=lambda t: int(t.get('gt_page_number', -1)))
        final_procedural_tasks.extend(tasks)

    logging.info(
        f"Aggregated and sorted {len(final_general_tasks)} general tasks and {len(final_procedural_tasks)} procedural tasks.")

    general_cols = ['doc_name', 'question_id', 'question_text', 'gt_answer_snippet', 'gt_page_number']
    procedural_cols = ['doc_name', 'question_id', 'question_text', 'gt_answer_snippet', 'gt_page_number',
                       'parsed_steps']

    for rater in ["A", "B"]:
        logging.info(f"--- Preparing files for Rater {rater} ---")
        gen_path = output_dir / f"general_audit_{rater}.xlsx"
        proc_path = output_dir / f"procedural_audit_{rater}.xlsx"
        create_xlsx_audit_file(final_general_tasks, general_cols, GENERAL_CHOICES, gen_path)
        create_xlsx_audit_file(final_procedural_tasks, procedural_cols, PROCEDURAL_CHOICES, proc_path)


if __name__ == "__main__":
    main()