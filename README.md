# A Semi-Automated Pipeline for Technical Manual QA Dataset Generation

This repository contains the official source code for the CIKM 2025 paper: "A Semi-Automated Pipeline for Synthetic Generation of Grounded, Structurally-Aware QA Datasets for Technical Manuals".

Our work presents a novel, semi-automated pipeline that demonstrates how to effectively use long-context Large Language Models (LLMs) for synthetic data generation with an integrated human-in-the-loop validation process. The pipeline processes entire technical manuals at once to generate a diverse set of high-quality questions, which are then refined through a rigorous automated filtering cascade and validated by human annotators to ensure high data quality and factual grounding.

## Key Features

- **End-to-End Pipeline**: From PDF parsing to a final, validated QA dataset.
- **Long-Context LLM Strategy**: Processes entire manuals in a single prompt, avoiding complex chunking and retrieval strategies.
- **Multi-Stage Filtering Cascade**: Ensures data quality through semantic deduplication, RAGAS-based grounding checks, and an LLM-as-judge.
- **Structurally-Aware Generation**: Produces a principled taxonomy of General, Procedural, and LLM-verified Unanswerable questions.
- **Reproducibility Focused**: Includes scripts to reproduce not only the pipeline but also the analysis and figures presented in the paper.

## Repository Structure

```
.
├── src/                      # All Python scripts for the pipeline and analysis
├── data/                     # Local-only directory for inputs and outputs (add to .gitignore)
│   ├── pdfs/                 # Where you will save the downloaded source manuals
│   └── processed/            # Where intermediate files from the pipeline are stored
├── prompts/                  # Contains the master prompt for QA generation
├── config/                   # The main settings.yaml configuration file
├── human_validation/         # Holds the Excel sheets for human audit tasks
├── paper_artifacts/          # Output folder for generated figures and tables
├── release/                  # Output folder for the final Zenodo archive
├── .env                      # Local file for API keys (ignored by Git)
├── .gitignore                # Specifies files and directories to be ignored by Git
├── LICENSE                   # MIT License for the code
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Setup and Installation

Follow these steps to set up your local environment.

### 1. Clone the Repository

```bash
git clone https://github.com/tduricic/cikm2025-techmanualqa.git
cd cikm2025-techmanualqa
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies

Install all required Python libraries from the requirements.txt file.

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

The pipeline requires API keys for Google (for Gemini models) and OpenAI (for the GPT-4 judge).

1. Create a file named `.env` in the root of the project directory.
2. Add your keys to this file as follows:

```bash
# .env file
# Get your Google API key from Google AI Studio
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"

# Get your OpenAI API key from the OpenAI platform
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
```

**Important**: The `.env` file is listed in `.gitignore` and should never be committed to version control.

## How to Run the Data Generation Pipeline

This is a step-by-step guide to generating the dataset from scratch.

### Step 0: Download Source Manuals

This pipeline operates on a set of 10 technical manuals. You must download them first.

1. Create the data directory:

```bash
mkdir -p data/pdfs
```

2. Download each of the following manuals and save them into the `data/pdfs/` directory.

| Manual Name | Filename | Source URL |
|---|---|---|
| Bosch Oven | bosch_oven.pdf | [Link](https://www.bosch-home.com/) |
| DeWalt Saw | dewalt_saw.pdf | [Link](https://www.dewalt.com/) |
| Dyson v12 Vacuum | dyson_v12.pdf | [Link](https://www.dyson.com/) |
| Electrolux Oven | electrolux_oven.pdf | [Link](https://www.electrolux.com/) |
| Hilti Hammer Drill | hilti_hammer.pdf | [Link](https://www.hilti.com/) |
| Makita Drill | makita_drill.pdf | [Link](https://www.makita.com/) |
| Miele Dishwasher | miele_dishwasher.pdf | [Link](https://www.miele.com/) |
| Miele Heat Pump Dryer | heat_pump_dryer.pdf | [Link](https://www.miele.com/) |
| Omron BP Monitor | omron_monitor.pdf | [Link](https://www.omron.com/) |
| Prusa MK4 3D Printer | prusa_3d-printer.pdf | [Link](https://www.prusa3d.com/) |

### Step 1: Parse PDFs into Markdown

Run the parsing script for each PDF. This converts the PDF into a structured `.jsonl` file that the pipeline can use.

```bash
# Example for one manual
python src/parse_pdf.py data/pdfs/bosch_oven.pdf

# Run this command for all 10 PDFs
```

The output will be saved in `data/processed/[manual_name]/`.

### Step 2: Generate and Filter QA Candidates

Run the main generation script for each parsed manual. This script performs the LLM generation and the entire automated filtering cascade.

```bash
# Example for one manual
python src/02_generate_candidates.py -i data/processed/bosch_oven/bosch_oven_pages.jsonl

# Run this command for all 10 processed manuals
```

This script saves the final selected candidates (`_candidates.jsonl`), run statistics (`_stats.json`), and the initial audit files (`_audit_*.jsonl`) in the same processed directory.

### Step 3: Prepare Human Audit Files

Aggregate the audit files from all manuals and create the user-friendly Excel spreadsheets for the annotators.

```bash
python src/03_prepare_audit_xlsx.py
```

This will create four `.xlsx` files inside the `human_validation/audit_sheets/` directory.

### Step 4: Perform Human Annotation (Manual Step)

This is a manual step. Provide the generated `.xlsx` files to your human annotators and have them fill them out according to the instructions contained within the spreadsheets.

### Step 5: Package the Final Dataset

Once the audit sheets are filled out, run the final packaging script. This script merges the human annotations with the candidate data and creates a clean, distributable archive ready for upload to Zenodo.

```bash
python src/08_create_zenodo_archive.py
```

The final archive (`TechManualQA_350.zip`) will be saved in the `release/` directory.

## Reproducing Paper Artifacts

You can regenerate the figures and tables from the paper by running the analysis scripts. These scripts should be run after you have completed the full data generation pipeline (Steps 1 and 2 above) and human annotation (Step 4).

```bash
# Generate Figure: Prompt Token vs Page Count
python src/04_create_input_stats_plot.py

# Generate Table: Filtering Cascade Statistics
python src/05_create_filtering_table.py

# Generate Figure: Score Distributions
python src/06_create_score_visualizations.py

# Generate Table: Human Audit Results
# (Requires the filled-out .xlsx files in human_validation/audit_sheets/)
python src/07_analyze_human_audit.py
```

The output figures and tables will be saved in the `paper_artifacts/` directory.

## Dataset Access

The final, complete TechManualQA-350 dataset, containing 350 questions with all associated metadata and human validation results, is permanently archived and publicly available on Zenodo.

Access the dataset at: https://doi.org/10.5281/zenodo.15689495

## Citation

If you use our pipeline, code, or dataset in your research, please cite our paper:

```bibtex
@inproceedings{duricic2025techmanualqa,
  author    = {Duricic, Tomislav and ElSayed, Neven and Kopeinik, Simone and Helic, Denis and Kowald, Dominik and Veas, Eduardo},
  title     = {A Semi-Automated Pipeline for Synthetic Generation of Grounded, Structurally-Aware QA Datasets for Technical Manuals},
  booktitle = {TBD},
  year      = {TBD},
  publisher = {TBD},
  address   = {TBDa},
}
```

## Licenses

- The source code in this repository is licensed under the **MIT License**.
- The TechManualQA-350 dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0) License**.