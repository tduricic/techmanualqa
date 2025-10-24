# 02
import os
import csv
import json
import sys
import random
import argparse
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import ast  # For parsing LLM step output
import math

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Attempt to import necessary libraries
try:
    import pandas as pd
    from sentence_transformers import SentenceTransformer, util
    from dotenv import load_dotenv
    from datasets import Dataset, Features, Value, Sequence
    from ragas import evaluate as ragas_eval
    from ragas.metrics import faithfulness, answer_correctness
    from sklearn.metrics import cohen_kappa_score
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please ensure prerequisites are installed:")
    print("pip install PyYAML sentence-transformers pandas scikit-learn datasets ragas python-dotenv")
    sys.exit(1)

# Import the unified LLM client
try:
    from techmanualqa.llm_client import create_llm_client, UnifiedLLMClient
except ImportError as e:
    print(f"Error importing llm_client: {e}")
    print("Please ensure llm_client.py is in the same directory or in your Python path.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(lineno)d: %(message)s')

# ------------------ CONFIGURATION LOADING ------------------------------------
try:
    import yaml

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parents[1]
    CONFIG_PATH = PROJECT_DIR / "config" / "settings.yaml"
    logging.info(f"Loading configuration from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully.")
except FileNotFoundError:
    logging.error(f"Configuration file not found at calculated path: {CONFIG_PATH}")
    logging.error("Ensure 'config/settings.yaml' exists relative to the script's parent directory.")
    sys.exit(1)
except ImportError:
    logging.error("PyYAML not found. Please install it (`pip install pyyaml`) to load config/settings.yaml.")
    sys.exit(1)
except yaml.YAMLError as e:
    logging.error(f"Error parsing configuration file {CONFIG_PATH}: {e}")
    sys.exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred during configuration loading: {e}", exc_info=True)
    sys.exit(1)

# --- Extract config values into global constants ---
try:
    # Pipeline Params
    OVERGEN_FACTOR = config['pipeline']['overgen_factor']
    DUP_THRESHOLD = config['pipeline']['dup_threshold']
    RAGAS_THRESHOLD = config['pipeline']['ragas_threshold']
    JUDGE_THRESHOLD = config['pipeline']['judge_threshold']
    FINAL_DATASET_SIZE = config['pipeline']['final_dataset_size']
    TARGET_RAW_ROWS = FINAL_DATASET_SIZE * OVERGEN_FACTOR

    # Models - now these are dictionaries with backend/model_name
    EMBED_MODEL = config['models']['embed']
    GENERATION_CONFIG = config['models']['generation']
    STEP_PARSING_CONFIG = config['models']['step_parsing']
    JUDGE_CONFIG = config['models']['judge']

    # Files
    MASTER_PROMPT_PATH = PROJECT_DIR / config['files']['master_prompt']
    RAW_OUTPUT_SUFFIX = config['files']['raw_output_suffix']
    CANDIDATE_SUFFIX = config['files']['candidate_suffix']
    STATS_SUFFIX = config['files']['stats_suffix']
    AUDIT_SUFFIX_A = config['files']['audit_suffix_a']
    AUDIT_SUFFIX_B = config['files']['audit_suffix_b']
    GOLD_DATASET_SUFFIX = config['files']['gold_dataset_suffix']

    # Quotas
    CATEGORY_TARGETS = config['quotas']['category_targets']
    if sum(CATEGORY_TARGETS.values()) != FINAL_DATASET_SIZE:
        logging.warning(
            f"Sum of CATEGORY_TARGETS ({sum(CATEGORY_TARGETS.values())}) != FINAL_DATASET_SIZE ({FINAL_DATASET_SIZE}). Using sum value.")
        FINAL_DATASET_SIZE = sum(CATEGORY_TARGETS.values())

    # Audit
    AUDIT_FRACTION = config['audit']['audit_fraction']

    # API Params
    JUDGE_DELAY_SECONDS = config['api_params']['judge_delay_seconds']
    MAX_PARSE_RETRIES = config['api_params']['max_parse_retries']
    RETRY_DELAY_SECONDS = config['api_params']['retry_delay_seconds']

except KeyError as e:
    logging.error(f"Missing key in configuration file {CONFIG_PATH}: {e}")
    sys.exit(1)

# --- Constants defined in script ---
PROCEDURAL_CATEGORY_NAME = "Procedural Step Inquiry"
LOCATION_DEF_CATEGORY_NAME = "Location/Definition"
UNANSWERABLE_CATEGORY_NAME = "Unanswerable"
SCHEMA = ["question_id", "persona", "doc_id", "language",
          "question_text", "category", "gt_answer_snippet",
          "gt_page_number", "_self_grounded"]
FINAL_SCHEMA = SCHEMA + ['parsed_steps', 'passed_strict_check', 'corrected_steps', 'procedural_comments']
SENTINEL_ROWS_INFO = [
    ({"question_id": "SENTINEL_BAD_01", "persona": "Technician", "doc_id": "N/A", "language": "fr",
      "question_text": "Explain about safety features thing?", "category": "Specification Lookup",
      "gt_answer_snippet": "Safety important.", "gt_page_number": "-1",
      "_self_grounded": "False", "parsed_steps": None, "passed_strict_check": False},
     {"answer_correct?": "no", "grounded?": "no", "question_clear?": "no"})
]
NUM_SENTINELS = len(SENTINEL_ROWS_INFO)

VALID_PERSONAS = ["Novice User", "Technician", "SafetyOfficer"]
VALID_CATEGORIES = [
    "Specification Lookup",
    "Tool/Material Identification",
    PROCEDURAL_CATEGORY_NAME,
    LOCATION_DEF_CATEGORY_NAME,
    "Conditional Logic/Causal Reasoning",
    "Safety Information Lookup",
    UNANSWERABLE_CATEGORY_NAME
]

# ------------------ INITIALIZE MODELS ----------------------------------------
print("Loading environment variables from .env file if present...")
load_dotenv()

try:
    # Check for required API keys based on config
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    litellm_api_key = os.getenv("LITELLM_API_KEY")

    # Validate that required keys are present
    required_backends = set()
    required_backends.add(GENERATION_CONFIG['backend'])
    required_backends.add(STEP_PARSING_CONFIG['backend'])
    required_backends.add(JUDGE_CONFIG['backend'])

    for backend in required_backends:
        if backend == 'google' and not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment but required by config")
        elif backend == 'openai' and not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment but required by config")
        elif backend == 'litellm' and not litellm_api_key:
            raise ValueError("LITELLM_API_KEY not found in environment but required by config")

    print("API keys validated successfully.")

    # Initialize LLM clients
    print(f"Initializing generation model: {GENERATION_CONFIG['backend']}/{GENERATION_CONFIG['model_name']}")
    generation_client = create_llm_client(GENERATION_CONFIG)

    print(f"Initializing step parsing model: {STEP_PARSING_CONFIG['backend']}/{STEP_PARSING_CONFIG['model_name']}")
    step_parsing_client = create_llm_client(STEP_PARSING_CONFIG)

    print(f"Initializing judge model: {JUDGE_CONFIG['backend']}/{JUDGE_CONFIG['model_name']}")
    judge_client = create_llm_client(JUDGE_CONFIG)

    print(f"Initializing embedding model: {EMBED_MODEL}")
    embedding_model = SentenceTransformer(EMBED_MODEL)

    print("Models initialized successfully.")

except ValueError as ve:
    logging.error(f"Configuration Error: {ve}")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error initializing models: {e}", exc_info=True)
    sys.exit(1)


# =============================================================================
# --- FUNCTION DEFINITIONS (Steps 1-10) ---
# =============================================================================

# ------------------ STEP 1: I/O UTILS --------------------------------------
def load_manual(jsonl_path_str: str) -> Tuple[List[Dict], str, str]:
    """Loads manual data from a JSONL file."""
    jsonl_path = Path(jsonl_path_str)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Input file not found at {jsonl_path_str}")
    pages = []
    print(f"Attempting to load manual from: {jsonl_path}")
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    page_data = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning(f"Skipping line {i + 1} invalid JSON.")
                    continue
                if "page_num" not in page_data or "markdown_content" not in page_data:
                    logging.warning(f"Line {i + 1} missing keys. Skipping.")
                    continue
                if not isinstance(page_data.get("markdown_content"), str):
                    page_data["markdown_content"] = str(page_data["markdown_content"])
                    logging.warning(f"Line {i + 1} content not string. Converted.")
                pages.append(page_data)
    except Exception as e:
        logging.error(f"Unexpected error loading {jsonl_path_str}: {e}", exc_info=True)
        sys.exit(1)
    if not pages:
        logging.error(f"No valid pages loaded from {jsonl_path_str}.")
        sys.exit(1)
    print(f"Successfully loaded {len(pages)} pages from {jsonl_path_str}")
    first_page = pages[0]
    doc_id = first_page.get("doc_id", "unknown_doc")
    language = first_page.get("language", "unknown_lang")
    print(f"Detected doc_id: {doc_id}, language: {language} (from first page)")
    return pages, doc_id, language


# ------------------ STEP 2a Helper: Prompt Building -------------------------
def build_prompt(pages: List[Dict]) -> str:
    """Loads the master prompt and inserts the formatted page data."""
    try:
        master_prompt_text = MASTER_PROMPT_PATH.read_text(encoding='utf-8')
    except Exception as e:
        logging.error(f"Error reading master prompt file {MASTER_PROMPT_PATH}: {e}", exc_info=True)
        sys.exit(1)
    pages_jsonl_string = "\n".join([json.dumps(p, ensure_ascii=False) for p in pages])
    placeholder = "<PASTE ALL JSONL LINES FOR THIS MANUAL HERE>"
    if placeholder not in master_prompt_text:
        logging.error(f"Placeholder '{placeholder}' not found in {MASTER_PROMPT_PATH}")
        sys.exit(1)
    full_prompt = master_prompt_text.replace(placeholder, pages_jsonl_string)
    return full_prompt


# ------------------ STEP 2a Helper: Generation ------------------------------
def over_generate(prompt: str, k: int, generation_client: UnifiedLLMClient) -> List[str]:
    """Calls the generation model k times to over-generate raw CSV rows."""
    all_raw_rows = []
    print(
        f"\n--- Starting Generation (x{k} calls, Temp={generation_client.temperature}, Target Lines ~{TARGET_RAW_ROWS}) ---")
    for i in range(k):
        print(f"Generation call {i + 1} of {k}...")
        try:
            raw_text = generation_client.generate(prompt)
            if raw_text:
                generated_lines = [line for line in raw_text.splitlines() if line.strip()]
                print(f"  -> Received {len(generated_lines)} non-empty lines.")
                all_raw_rows.extend(generated_lines)
            else:
                logging.warning(f"Call {i + 1} did not produce expected text output.")
        except Exception as e:
            if generation_client.is_rate_limit_error(e):
                logging.warning(f"Rate limit hit on generation call {i + 1}. Waiting {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            logging.error(f"Error during generation call {i + 1}: {e}", exc_info=True)
            print("  -> Attempting to continue...")
    print(f"--- Generation Complete: Collected {len(all_raw_rows)} raw rows total ---")
    return all_raw_rows


# ------------------ STEP 3: PARSING -----------------------------------------
def parse_rows(raw_rows: List[str], schema: List[str], doc_id: str, language: str) -> pd.DataFrame:
    """Parses raw CSV strings into DataFrame, initializes parsed_steps."""
    parsed_data = []
    expected_columns = len(schema)
    row_num_counter = 0
    doc_prefix = doc_id.split('.')[0] if '.' in doc_id else doc_id
    print(f"\n--- Starting Parsing of {len(raw_rows)} raw rows ---")
    for i, raw_row in enumerate(raw_rows):
        try:
            reader = csv.reader([raw_row], quotechar='"', doublequote=True, skipinitialspace=True)
            cells = next(reader)
            if len(cells) != expected_columns:
                logging.warning(f"Row {i + 1} bad columns ({len(cells)}/{expected_columns}). Skipping.")
                continue
            row_dict = dict(zip(schema, cells))
            row_dict = {k: v.strip() if isinstance(v, str) else v for k, v in row_dict.items()}
            row_dict['doc_id'] = doc_id
            row_dict['language'] = language
            row_num_counter += 1
            row_dict['question_id'] = f"{doc_prefix}_Q{row_num_counter:03d}"

            if str(row_dict.get('_self_grounded')).strip().capitalize() not in ["True", "False"]:
                row_dict['_self_grounded'] = "False"
            else:
                row_dict['_self_grounded'] = str(row_dict.get('_self_grounded')).strip().capitalize()

            if row_dict.get('gt_page_number') != "None":
                try:
                    int(row_dict['gt_page_number'])
                except (ValueError, TypeError):
                    logging.warning(
                        f"Row {i + 1} QID {row_dict['question_id']} invalid page number '{row_dict['gt_page_number']}'. Setting to None.")
                    row_dict['gt_page_number'] = "None"

            if 'persona' not in row_dict:
                logging.warning(
                    f"Row {i + 1} QID {row_dict['question_id']} missing 'persona'. Setting to empty string.")
                row_dict['persona'] = ""
            if 'category' not in row_dict:
                logging.warning(
                    f"Row {i + 1} QID {row_dict['question_id']} missing 'category'. Setting to empty string.")
                row_dict['category'] = ""

            row_dict['parsed_steps'] = None
            parsed_data.append(row_dict)
        except csv.Error as csv_e:
            logging.warning(f"CSV parsing error row {i + 1}: {csv_e}. Skipping.")
        except Exception as e:
            logging.warning(f"General parsing error row {i + 1}: {e}. Skipping.", exc_info=True)
            continue
    print(f"--- Parsing Complete: Successfully parsed {len(parsed_data)} rows ---")
    if not parsed_data:
        logging.error("No rows parsed.")
        return pd.DataFrame()
    df = pd.DataFrame(parsed_data)
    return df


# --- Step 3a: Filter Invalid Persona/Category ---
def filter_invalid_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Filters out rows with invalid persona or category values."""
    if df.empty:
        logging.info("Filter Invalid Metadata: Skipping, input DataFrame is empty.")
        return df

    if 'persona' not in df.columns or 'category' not in df.columns:
        logging.error("Filter Invalid Metadata: Missing 'persona' or 'category' column. Skipping filter.")
        return df

    n_rows_before = len(df)
    print(
        f"\n--- Step 3a: Filtering Invalid Persona/Category (Valid Personas: {len(VALID_PERSONAS)}, Valid Categories: {len(VALID_CATEGORIES)}) ---")

    valid_persona_mask = df['persona'].isin(VALID_PERSONAS)
    valid_category_mask = df['category'].isin(VALID_CATEGORIES)
    combined_mask = valid_persona_mask & valid_category_mask

    invalid_rows = df[~combined_mask]
    if not invalid_rows.empty:
        invalid_personas_found = invalid_rows[~valid_persona_mask]['persona'].unique()
        invalid_categories_found = invalid_rows[~valid_category_mask]['category'].unique()
        logging.warning(f"Found {len(invalid_rows)} rows with invalid metadata:")
        if len(invalid_personas_found) > 0:
            logging.warning(f"  - Invalid Personas encountered: {list(invalid_personas_found)}")
        if len(invalid_categories_found) > 0:
            logging.warning(f"  - Invalid Categories encountered: {list(invalid_categories_found)}")

    df_filtered = df[combined_mask].reset_index(drop=True)
    n_rows_after = len(df_filtered)
    n_removed = n_rows_before - n_rows_after

    print(f"Filter Invalid Metadata: Kept {n_rows_after} / {n_rows_before} rows (Removed {n_removed}).")
    print("\n--- Step 3a Complete (Invalid Metadata Filter) ---")
    return df_filtered


# --- Step 3b: Add Parsed Steps via LLM ---
def llm_parse_snippet_to_steps(snippet: str, parsing_client: UnifiedLLMClient, max_retries=MAX_PARSE_RETRIES) -> List[
    str]:
    """
    Uses the specified LLM client to parse a text snippet into a list of procedural steps.
    Returns None if parsing fails or yields no steps.
    """
    if not snippet or not isinstance(snippet, str):
        return None

    prompt = f"""Parse the following text, which represents procedural steps from a technical manual, into a list of distinct steps. A distinct step typically represents a single complete action or instruction.

**Instructions:**
1. Preserve the meaning and approximate number of steps accurately.
2. Maintain the original wording within each step as much as possible.
3. Treat text separated by ellipses ('...') as part of the SAME step unless a new action clearly begins after the ellipses. Merge the related text.
4. Try to logically rejoin words that might be hyphenated across line breaks in the input text.
5. Ignore non-standard bullet points or markers (like '', '–') at the beginning of lines when determining step content, focus on the actions described.
6. Output ONLY a single, valid, Python-style list of strings (e.g., ["Step 1...", "Step 2..."]). Do not include any other text, explanations, or markdown formatting.

--- Examples ---

Example 1:
Text to Parse:
```
 Pull the upper filter out.
 Remove fluff . . .
. . . from surfaces and de-
flector.
 Push filter back.
```
Parsed List:
["Pull the upper filter out.", "Remove fluff from surfaces and deflector.", "Push filter back."]

Example 2:
Text to Parse:
```
1. Attach tool.
2. Unscrew anti-clockwise.
3. Insert new part. Note: Ensure
correct orientation.
4. Screw clockwise.
```
Parsed List:
["Attach tool.", "Unscrew anti-clockwise.", "Insert new part. Note: Ensure correct orientation.", "Screw clockwise."]

Example 3:
Text to Parse:
```
Remove any fluff after every drying\ncycle.\n\n Pull the upper fluff filter forwards to\nremove it.\n\n Remove the fluff (see arrows) . . .\n\n . . . from the surfaces of all the fluff\n\nfilters.\n\n . . . from the perforated laundry de‐\nflector.\n\n Push the upper fluff filter back into\nposition until it clicks.
```
Parsed List:
["Remove any fluff after every drying cycle.", "Pull the upper fluff filter forwards to remove it.", "Remove the fluff (see arrows) from the surfaces of all the fluff filters and from the perforated laundry deflector.", "Push the upper fluff filter back into position until it clicks."]

--- Task ---

Text to Parse:
    {snippet}
Parsed List:
"""

    for attempt in range(max_retries):
        try:
            logging.debug(f"Parsing attempt {attempt + 1}/{max_retries} using {parsing_client.model_name}")

            raw_text = parsing_client.generate(prompt, max_tokens=3000)

            # Check if we got any response at all
            if not raw_text:
                logging.warning(f"LLM Parser call returned no text on attempt {attempt + 1}.")
                if attempt < max_retries - 1:
                    logging.warning(f"LLM Parsing failed attempt {attempt + 1}. Retrying...")
                    time.sleep(1)
                    continue
                else:
                    logging.error(f"LLM Parsing failed after {max_retries} attempts (no text returned).")
                    return None

            logging.debug(f"LLM Parser Raw Response (Attempt {attempt + 1}): '{raw_text[:200]}...'")

            # Process response
            try:
                # Remove markdown code blocks if present
                cleaned_text = re.sub(r"^```python\s*|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
                logging.debug(f"LLM Parser Cleaned Text (first 200 chars): '{cleaned_text[:200]}...'")

                parsed_list = None

                # Check if it's a valid list format
                if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
                    logging.debug("Attempting ast.literal_eval...")
                    try:
                        parsed_list = ast.literal_eval(cleaned_text)
                    except (SyntaxError, ValueError) as e:
                        logging.warning(f"ast.literal_eval failed: {e}. Response might be truncated.")
                        # Check if truncated (doesn't end with ])
                        if not cleaned_text.rstrip().endswith(']'):
                            logging.warning("Response appears truncated (no closing bracket). Increase max_tokens.")
                        parsed_list = None
                else:
                    logging.warning(f"LLM Parser output not list format: '{cleaned_text[:100]}...' Trying line split.")
                    # Fallback to line splitting
                    parsed_list = [line.strip() for line in cleaned_text.splitlines() if line.strip()]

                # Validate the parsed result
                if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                    final_steps = [step.strip() for step in parsed_list if step.strip()]
                    if final_steps:
                        logging.debug(f"LLM Parser SUCCESS -> Parsed {len(final_steps)} steps")
                        return final_steps
                    else:
                        logging.warning(f"LLM Parser parsed to EMPTY list: '{cleaned_text[:100]}...'")
                else:
                    logging.warning(f"LLM Parser output not list of strings: Type={type(parsed_list)}")

            except Exception as parse_error:
                logging.warning(f"LLM Parser output eval failed: {parse_error}. Cleaned Text: '{cleaned_text[:200]}'")

            # If we get here, parsing failed for this attempt
            if attempt < max_retries - 1:
                logging.warning(f"LLM Parsing failed attempt {attempt + 1}. Retrying...")
                time.sleep(1)
                continue
            else:
                logging.error(f"LLM Parsing failed after {max_retries} attempts.")
                return None

        except Exception as e:
            if parsing_client.is_rate_limit_error(e):
                logging.warning(f"Rate limit hit... Waiting {RETRY_DELAY_SECONDS}s...")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    logging.error(f"Rate limit persists. Skipping snippet.")
                    return None
            logging.error(f"API Error LLM parsing: {e}", exc_info=True)
            return None

    logging.error(f"LLM Parsing failed all retries.")
    return None


def add_parsed_steps_llm(df: pd.DataFrame, parsing_client: UnifiedLLMClient) -> pd.DataFrame:
    """Adds/Updates 'parsed_steps' column by parsing procedural snippets using an LLM."""
    if df.empty:
        logging.info("Add Parsed Steps: Skipping, input empty.")
        return df.assign(parsed_steps=None)
    if 'category' not in df.columns or 'gt_answer_snippet' not in df.columns:
        logging.error("Missing columns for step parsing.")
        return df.assign(parsed_steps=None)
    print(f"\n--- Step 3b: Parsing Procedural Snippets using LLM ({parsing_client.model_name}) ---")
    parsed_steps_col_data = []
    rows_to_parse_indices = df[df['category'] == PROCEDURAL_CATEGORY_NAME].index
    num_to_parse = len(rows_to_parse_indices)
    print(f"Found {num_to_parse} rows in category '{PROCEDURAL_CATEGORY_NAME}' to parse.")
    parsed_count = 0
    for index, row in df.iterrows():
        if index in rows_to_parse_indices:
            snippet = row.get('gt_answer_snippet')
            if isinstance(snippet, str) and snippet:
                parsed_list = llm_parse_snippet_to_steps(snippet, parsing_client)
                parsed_steps_col_data.append(parsed_list)
                if parsed_list is not None:
                    parsed_count += 1
                time.sleep(0.1)
            else:
                logging.warning(f"QID {row.get('question_id')} procedural bad snippet.")
                parsed_steps_col_data.append(None)
        else:
            parsed_steps_col_data.append(None)

    if len(parsed_steps_col_data) != len(df):
        logging.error(
            f"Parsed steps list length ({len(parsed_steps_col_data)}) != DataFrame length ({len(df)}). Cannot assign column.")
        return df
    df['parsed_steps'] = parsed_steps_col_data
    print(f"LLM Step Parsing complete. Successfully parsed steps for {parsed_count} / {num_to_parse} rows.")
    df['parsed_steps'] = df['parsed_steps'].astype(object)
    print("\n--- Step 3b Complete (LLM Step Parsing) ---")
    return df


# ------------------ STEP 4: Deduplication Filter -----------------------------
def deduplicate(df: pd.DataFrame, embedder: SentenceTransformer, threshold: float, model_name_str: str) -> Tuple[
    pd.DataFrame, List[float]]:
    """
    Removes rows with questions too similar to preceding questions and returns all
    unique pairwise similarity scores for analysis.
    """
    if df.empty or 'question_text' not in df.columns:
        logging.info("Deduplication: Skipping, DataFrame is empty.")
        return df, []

    questions = df["question_text"].tolist()
    n_rows_before = len(df)
    print(f"\n--- Step 4: Deduplicating Questions ---")
    print(f"Deduplication: Encoding {n_rows_before} questions using '{model_name_str}'...")
    embeddings = embedder.encode(questions, normalize_embeddings=True, show_progress_bar=True)

    all_similarity_scores = []
    try:
        print("Deduplication: Calculating similarity score distribution...")
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        import numpy as np
        upper_triangle_indices = np.triu_indices(len(questions), k=1)
        all_similarity_scores = similarity_matrix[upper_triangle_indices].tolist()
    except ImportError:
        logging.warning("Numpy is required to calculate similarity distribution. Please install it.")
    except Exception as e:
        logging.warning(f"An error occurred during similarity score calculation: {e}")

    indices_to_keep = []
    print(f"Deduplication: Filtering questions with threshold {threshold:.2f}...")
    for i, emb_i in enumerate(embeddings):
        is_duplicate = False
        for j in indices_to_keep:
            emb_j = embeddings[j]
            similarity = util.cos_sim(emb_i, emb_j).item()
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            indices_to_keep.append(i)

    deduplicated_df = df.iloc[indices_to_keep].reset_index(drop=True)
    n_rows_after = len(deduplicated_df)
    n_removed = n_rows_before - n_rows_after
    print(f"Deduplication: Kept {n_rows_after} out of {n_rows_before} rows (Removed {n_removed}).")
    print(f" Shape after Dedupe: {deduplicated_df.shape}")
    print("\n--- Step 4 Complete ---")

    return deduplicated_df, all_similarity_scores


# ------------------ STEP 5: Page Check Annotation ----------------------------
def page_check_annotate(df: pd.DataFrame, pages_data: List[Dict]) -> pd.DataFrame:
    """Annotates DataFrame with 'passed_strict_check' boolean column."""
    if df.empty:
        logging.info("Page Check Annotate: Skipping.")
        df['passed_strict_check'] = pd.Series(dtype='boolean')
        return df
    if not pages_data:
        logging.warning("Page Check Annotate: Missing pages_data.")
        df['passed_strict_check'] = False
        return df
    print(f"\n--- Step 5: Annotating Snippet Grounding ---")
    print(f"Page Check Annotate: Verifying strict snippet presence for {len(df)} rows...")
    page_lookup = {page['page_num']: page.get('markdown_content', '') for page in pages_data if
                   isinstance(page.get('page_num'), int) and isinstance(page.get('markdown_content'), str)}
    if not page_lookup:
        logging.warning("Page Check Annotate: Failed lookup creation.")
        df['passed_strict_check'] = False
        return df
    passed_strict_list = []
    for index, row in df.iterrows():
        passed = False
        category = row.get('category')
        snippet = row.get('gt_answer_snippet')
        page_num_str = row.get('gt_page_number')
        if category == UNANSWERABLE_CATEGORY_NAME or page_num_str == "None":
            passed = True
        else:
            if isinstance(snippet, str) and snippet and page_num_str != "None":
                try:
                    page_num_int = int(page_num_str)
                    page_content = page_lookup.get(page_num_int)
                    if page_content is not None and isinstance(page_content, str) and snippet in page_content:
                        passed = True
                except (ValueError, TypeError):
                    pass
        passed_strict_list.append(passed)

    if len(passed_strict_list) != len(df):
        logging.error(
            f"Strict check list length ({len(passed_strict_list)}) != DataFrame length ({len(df)}). Cannot assign column.")
        df['passed_strict_check'] = pd.Series(dtype='boolean')
    else:
        df['passed_strict_check'] = pd.Series(passed_strict_list, index=df.index, dtype='boolean')

    n_passed = df['passed_strict_check'].sum()
    n_failed = len(df) - n_passed
    print(f"Page Check Annotate: Marked {n_passed} rows pass strict check, {n_failed} fail.")
    print(f" Shape after Annotation: {df.shape}")
    print("\n--- Step 5 Complete ---")
    return df


# ------------------ STEP 6: RAGAS Filter ------------------------------------
def add_ragas_scores(df: pd.DataFrame, pages_data: List[Dict]) -> Tuple[
    pd.DataFrame, List[float], List[float]]:
    """
    Adds Ragas scores to the DataFrame but does NOT filter.
    Returns the full DataFrame with new score columns and lists of scores.
    """
    if 'ragas_eval' not in globals() or 'faithfulness' not in globals() or 'answer_correctness' not in globals() or 'Dataset' not in globals():
        logging.warning("Ragas/Datasets components not available. Skipping RAGAS scoring.")
        df['ragas_faithfulness'] = -1.0
        df['ragas_correctness'] = -1.0
        return df, [], []
    if df.empty:
        logging.info("Ragas Score: Skipping, DataFrame is empty.")
        df['ragas_faithfulness'] = -1.0
        df['ragas_correctness'] = -1.0
        return df, [], []
    if not pages_data:
        logging.warning("Ragas Score: Missing pages_data. Skipping.")
        df['ragas_faithfulness'] = -1.0
        df['ragas_correctness'] = -1.0
        return df, [], []

    n_rows_before = len(df)
    print(f"\n--- Step 6: Scoring with Ragas ---")
    print(f"Ragas Score: Preparing data for {n_rows_before} rows...")
    page_lookup = {page['page_num']: page.get('markdown_content', '') for page in pages_data if
                   isinstance(page.get('page_num'), int) and isinstance(page.get('markdown_content'), str)}
    if not page_lookup:
        logging.warning("Ragas Score: Failed page lookup creation. Skipping.")
        df['ragas_faithfulness'] = -1.0
        df['ragas_correctness'] = -1.0
        return df, [], []

    ragas_data_list = []
    original_indices = []
    # Iterate over all rows, not just non-unanswerable, to align indices
    for index, row in df.iterrows():
        # Only prepare rows that are answerable and valid
        if row.get('category') != UNANSWERABLE_CATEGORY_NAME:
            question = row.get('question_text')
            answer = row.get('gt_answer_snippet')
            page_num_str = row.get('gt_page_number')
            if not all([isinstance(q, str) and q for q in [question, answer]]) or page_num_str == "None":
                continue
            try:
                page_num_int = int(page_num_str)
                page_content = page_lookup.get(page_num_int)
                if page_content and isinstance(page_content, str):
                    ragas_data_list.append(
                        {"question": question, "answer": answer, "contexts": [page_content], "ground_truth": answer})
                    original_indices.append(index) # This is key
            except (ValueError, TypeError):
                continue

    if not ragas_data_list:
        logging.warning("Ragas Score: No valid data to evaluate. Adding empty score columns.")
        df['ragas_faithfulness'] = -1.0
        df['ragas_correctness'] = -1.0
        return df, [], []

    print(f"Ragas Score: Prepared {len(ragas_data_list)} rows for evaluation.")
    ragas_features = Features(
        {'question': Value('string'), 'answer': Value('string'), 'contexts': Sequence(Value('string')),
         'ground_truth': Value('string')})
    eval_dataset = Dataset.from_list(ragas_data_list, features=ragas_features)
    metrics = [faithfulness, answer_correctness]
    scores_df = None

    try:
        print(
            "Running Ragas evaluation (faithfulness, answer_correctness)... This may take time and requires API calls.")
        results = ragas_eval(dataset=eval_dataset, metrics=metrics, raise_exceptions=False)
        scores_df = results.to_pandas()
    except Exception as e:
        logging.error(f"Ragas evaluation failed: {e}", exc_info=True)
        logging.warning("Ragas Score: Skipping RAGAS scoring due to error.")
        df['ragas_faithfulness'] = -1.0
        df['ragas_correctness'] = -1.0
        return df, [], []

    if scores_df is None or scores_df.empty:
        logging.warning("Ragas Score: No scores returned. Adding empty score columns.")
        df['ragas_faithfulness'] = -1.0
        df['ragas_correctness'] = -1.0
        return df, [], []

    scores_df = scores_df.fillna(0.0)
    faithfulness_scores_list = scores_df['faithfulness'].tolist()
    correctness_scores_list = scores_df['answer_correctness'].tolist()

    scores_to_merge = pd.DataFrame({
        'ragas_faithfulness': scores_df['faithfulness'],
        'ragas_correctness': scores_df['answer_correctness'],
        'original_index': original_indices
    }).set_index('original_index')

    df = df.join(scores_to_merge) # Add scores

    # Fill non-evaluated rows (like Unanswerable) with -1.0
    df['ragas_faithfulness'].fillna(-1.0, inplace=True)
    df['ragas_correctness'].fillna(-1.0, inplace=True)

    print(f"Ragas Score: Successfully added scores for {len(scores_df)} rows.")
    print(f" Shape after Ragas scoring: {df.shape}")
    print("\n--- Step 6 Complete (Scoring) ---")

    # *** CHANGED: Return the FULL DataFrame, not a filtered one ***
    return df, faithfulness_scores_list, correctness_scores_list


# ------------------ STEP 7: LLM Judge Scoring (Modified) -------------------
def llm_judge_score(question: str, answer_snippet: str, context: str, judge_client: UnifiedLLMClient) -> int:
    """Calls the specified LLM judge to score the QA pair based on context."""
    default_score = 1
    system_prompt = "You are a strict QA pair evaluator. You will be given a Question, an Answer Snippet supposedly extracted from a Context document, and the Context itself. Evaluate if the Answer Snippet is a good and correct answer to the Question, based *only* on the provided Context. Output ONLY a single integer score from 1 to 5 based on the following scale:"
    rubric = """
5: Snippet fully, correctly, and clearly answers the Question based *only* on the Context.
4: Snippet answers the Question correctly based on the Context, but might be slightly awkward, verbose, or contain very minor irrelevant info from the context.
3: Snippet partially answers the Question based on the Context or contains minor inaccuracies according to the Context.
2: Snippet is related to the Question but is significantly inaccurate or fails to answer the core of the Question based on the Context.
1: Snippet is irrelevant to the Question or completely wrong according to the Context.
"""
    user_prompt = f"Context:\n```\n{context}\n```\n\nQuestion:\n```\n{question}\n```\n\nAnswer Snippet:\n```\n{answer_snippet}\n```\n\nScore (1-5):"

    full_prompt = system_prompt + "\n" + rubric + "\n\n" + user_prompt

    try:
        content = judge_client.generate(full_prompt, max_tokens=5)
        match = re.search(r'\d', content)
        if match:
            score = int(match.group())
            return score if 1 <= score <= 5 else default_score
        else:
            logging.warning(f"Judge no digit ('{content}').")
            return default_score
    except Exception as e:
        if judge_client.is_rate_limit_error(e):
            logging.warning(f"Judge rate limit hit.")
            return default_score
        logging.error(f"Judge API error: {e}")
        return default_score


# Renamed from judge_filter and removed threshold argument
def add_judge_scores(df: pd.DataFrame, pages_data: List[Dict], judge_client: UnifiedLLMClient) -> Tuple[
    pd.DataFrame, List[int]]:
    """
    Adds scores from an LLM judge to the DataFrame but does NOT filter.
    Returns the full DataFrame with new score column and a list of scores.
    """
    if df.empty:
        logging.info("Judge Score: Skipping, DataFrame is empty.")
        df['judge_score'] = -1
        return df, []
    if not pages_data:
        logging.warning("Judge Score: Missing pages_data. Skipping.")
        df['judge_score'] = -1
        return df, []

    n_rows_before = len(df)
    print(f"\n--- Step 7: Scoring Answerable with LLM Judge ---")
    print(f"Judge Score: Preparing {n_rows_before} rows (model '{judge_client.model_name}')...")
    page_lookup = {page['page_num']: page.get('markdown_content', '') for page in pages_data if
                   isinstance(page.get('page_num'), int) and isinstance(page.get('markdown_content'), str)}
    if not page_lookup:
        logging.warning("Judge Score: Failed lookup creation. Skipping.")
        df['judge_score'] = -1
        return df, []

    scores = {}
    indices_to_judge = df[df['category'] != UNANSWERABLE_CATEGORY_NAME].index
    num_to_judge = len(indices_to_judge)
    print(f"Judge Score: Evaluating {num_to_judge} answerable rows...")

    for i, index in enumerate(indices_to_judge):
        if (i + 1) % 10 == 0:
            print(f"  Judging row {i + 1} of {num_to_judge}...")
        row = df.loc[index]
        question = row.get('question_text')
        answer_snippet = row.get('gt_answer_snippet')
        page_num_str = row.get('gt_page_number')
        if not all([isinstance(q, str) and q for q in [question, answer_snippet]]) or page_num_str == "None":
            scores[index] = 1 # Give a failing score
            continue
        try:
            page_num_int = int(page_num_str)
            page_content = page_lookup.get(page_num_int)
        except (ValueError, TypeError):
            scores[index] = 1 # Give a failing score
            continue
        if not isinstance(page_content, str) or not page_content:
            scores[index] = 1 # Give a failing score
            continue
        try:
            score = llm_judge_score(question, answer_snippet, page_content, judge_client)
            scores[index] = score
            time.sleep(JUDGE_DELAY_SECONDS)
        except Exception as e:
            logging.error(f"Judge Score error QID {row.get('question_id')}: {e}", exc_info=True)
            scores[index] = 1 # Give a failing score

    scores_series = pd.Series(scores)
    df['judge_score'] = scores_series
    df['judge_score'] = df['judge_score'].fillna(-1) # Fill non-evaluated (e.g., Unanswerable)
    df['judge_score'] = df['judge_score'].astype(int)

    judge_scores_list = list(scores.values()) # This is correct

    # *** CHANGED: Removed the filtering logic ***

    print(f"Judge Score: Kept {len(df)} / {n_rows_before} rows (Scoring only).")
    print(f" Shape after Judge Scoring: {df.shape}")
    print("\n--- Step 7 Complete (Scoring) ---")

    # *** CHANGED: Return the FULL DataFrame, not a filtered one ***
    return df, judge_scores_list


# ------------------ STEP 7b: Unanswerable Verification ----------------------
def get_full_manual_text(pages_data: List[Dict]) -> str:
    """Combines markdown content from all pages into a single string."""
    return "\n\n".join(
        [f"--- Page {p.get('page_num', 'N/A')} ---\n{p.get('markdown_content', '')}" for p in pages_data])


def verify_unanswerable(df_unanswerable: pd.DataFrame, full_manual_text: str,
                        judge_client: UnifiedLLMClient) -> pd.Index:
    """Uses the specified judge client to verify unanswerable questions."""
    if df_unanswerable.empty:
        return pd.Index([])
    print(f"\n--- Step 7b: Verifying Unanswerable Questions ---")
    print(f"Verifying {len(df_unanswerable)} unanswerable questions using judge model {judge_client.model_name}...")
    indices_confirmed_unanswerable = []
    unanswerable_confirmation_string = "Unanswerable"
    for index, row in df_unanswerable.iterrows():
        question = row.get('question_text')
        qid = row.get('question_id', 'N/A')
        if not isinstance(question, str) or not question:
            continue
        system_prompt = f"You are an assistant evaluating questions against a document. Based ONLY on the following document content, please answer the question concisely. If the document does not contain the information to answer the question, reply with the single word '{unanswerable_confirmation_string}' and nothing else."
        user_prompt = f"DOCUMENT CONTENT:\n```\n{full_manual_text}\n```\n\nQUESTION:\n```\n{question}\n```\n\nANSWER:"

        full_prompt = system_prompt + "\n\n" + user_prompt

        try:
            answer_text = judge_client.generate(full_prompt, max_tokens=100)
        except Exception as e:
            if judge_client.is_rate_limit_error(e):
                logging.warning(f"Verifier rate limit hit QID {qid}. Keeping.")
                indices_confirmed_unanswerable.append(index)
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            logging.error(f"Error verifying QID {qid}: {e}")
            indices_confirmed_unanswerable.append(index)
            time.sleep(JUDGE_DELAY_SECONDS)
            continue

        if answer_text:
            if answer_text.lower() == unanswerable_confirmation_string.lower():
                indices_confirmed_unanswerable.append(index)
            else:
                logging.info(f"QID {qid} 'Unanswerable' potentially found answer: '{answer_text[:100]}...'")
        else:
            logging.warning(f"Verifier no text output QID {qid}. Keeping row.")
            indices_confirmed_unanswerable.append(index)
        time.sleep(JUDGE_DELAY_SECONDS)

    print(f"Verification complete: Confirmed/Kept {len(indices_confirmed_unanswerable)} unanswerable rows.")
    print("\n--- Step 7b Complete ---")
    return pd.Index(indices_confirmed_unanswerable)


# ------------------ STEP 8: Quota Selection (Prioritized) ------------------
def quota_select(df: pd.DataFrame, category_targets: Dict[str, int], final_size: int, random_seed: int = 42) -> Tuple[
    pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Selects rows prioritizing 'passed_strict_check' == True. Returns df and details."""
    if df.empty:
        logging.error("Quota Selection: Input empty.")
        raise ValueError("Quota selection on empty DataFrame.")
    print(f"\n--- Step 8: Selecting Final Rows per Category Quota ---")
    print(f"Quota Selection: Selecting up to {final_size} rows, prioritizing strictly checked rows...")
    final_rows_list = []
    quota_details = {}
    all_quotas_met = True
    if 'category' not in df.columns:
        logging.error("Quota Selection: 'category' missing.")
        raise ValueError("'category' column missing.")
    has_strict_check_col = 'passed_strict_check' in df.columns
    if not has_strict_check_col:
        logging.warning("Quota Selection: 'passed_strict_check' column missing. Performing random selection only.")
    available_categories = set(df['category'].unique())
    for category_name in category_targets.keys():
        if category_name not in available_categories:
            logging.warning(f"Quota Selection: Target category '{category_name}' not found in available data.")
    for category_name, target_count in category_targets.items():
        print(f"  Processing Category: '{category_name}' (Target: {target_count})")
        df_pool = df[df['category'] == category_name].copy()
        available_total = len(df_pool)
        if available_total == 0:
            logging.warning(f"  -> Quota Warning: No rows available.")
            quota_details[category_name] = {'needed': target_count, 'available': 0, 'strict_taken': 0,
                                            'nonstrict_taken': 0, 'selected_total': 0}
            all_quotas_met = False
            continue
        selected_rows_for_cat_list = []
        strict_taken_count = 0
        nonstrict_taken_count = 0
        if has_strict_check_col:
            df_strict_passed = df_pool[df_pool['passed_strict_check'] == True]
            df_strict_failed = df_pool[df_pool['passed_strict_check'] == False]
            available_strict = len(df_strict_passed)
            available_failed = len(df_strict_failed)
            print(f"  -> Available: {available_strict} strict=True, {available_failed} strict=False.")
            num_needed = target_count
            num_to_take_strict = min(num_needed, available_strict)
            if num_to_take_strict > 0:
                selected_strict = df_strict_passed if available_strict <= num_to_take_strict else df_strict_passed.sample(
                    n=num_to_take_strict, random_state=random_seed)
                selected_rows_for_cat_list.extend(selected_strict.to_dict('records'))
                num_needed -= num_to_take_strict
                strict_taken_count = num_to_take_strict
                print(f"  -> Selected {strict_taken_count} strict.")
            if num_needed > 0 and available_failed > 0:
                num_to_take_failed = min(num_needed, available_failed)
                selected_failed = df_strict_failed if available_failed <= num_to_take_failed else df_strict_failed.sample(
                    n=num_to_take_failed, random_state=random_seed)
                selected_rows_for_cat_list.extend(selected_failed.to_dict('records'))
                num_needed -= num_to_take_failed
                nonstrict_taken_count = num_to_take_failed
                print(f"  -> Selected {nonstrict_taken_count} non-strict.")
            if num_needed > 0:
                logging.warning(f"  -> Quota Warning: Category '{category_name}' not fully met. Short by {num_needed}.")
                all_quotas_met = False
        else:
            num_to_select = min(target_count, available_total)
            print(f"  -> Selecting {num_to_select} randomly (no strict check data).")
            if available_total < target_count:
                logging.warning(f"  -> Quota Warning: Only {available_total} available.")
                all_quotas_met = False
            selected_rows_df = df_pool.sample(n=num_to_select, random_state=random_seed)
            selected_rows_for_cat_list.extend(selected_rows_df.to_dict('records'))
            nonstrict_taken_count = num_to_select
        quota_details[category_name] = {'needed': target_count, 'available': available_total,
                                        'strict_taken': strict_taken_count, 'nonstrict_taken': nonstrict_taken_count,
                                        'selected_total': strict_taken_count + nonstrict_taken_count}
        final_rows_list.extend(selected_rows_for_cat_list)
    final_df = pd.DataFrame(final_rows_list)
    if not final_df.empty:
        final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print(f"\nQuota Selection: Total rows selected = {len(final_df)}")
    quota_details['_summary'] = {'target_total': final_size, 'selected_total': len(final_df),
                                 'all_quotas_met': all_quotas_met}
    if len(final_df) != final_size:
        logging.error(
            f"Quota Failure: Final size ({len(final_df)}) != target ({final_size}). This usually means not enough valid rows survived the filtering steps.")
        logging.error(f"Quota Details:\n{json.dumps(quota_details, indent=2)}")
        logging.error("Proceeding with the rows selected, but final dataset size requirement NOT met.")
    else:
        print(f"Quota Selection: Successfully selected {len(final_df)} rows matching the target size.")
    print(f" Shape after Quota Select: {final_df.shape}")
    print("\n--- Step 8 Complete ---")
    return final_df, quota_details


# ------------------ STEP 9: Audit Preparation --------------------------------
def export_audit_slice(df: pd.DataFrame, output_base_path: Path, audit_fraction: float, random_seed: int = 42):
    """
    Exports candidates and audit/review files. The sample size for the general audit
    is determined by a fixed audit_fraction from the config.
    """
    if df.empty:
        logging.warning("Audit Export: Input DataFrame empty. Skipping export.")
        return 0, 0

    SENTINEL_PROCEDURAL_INFO = (
        {
            "question_id": "SENTINEL_PROC_BAD_01",
            "question_text": "How do I reboot the main server rack?",
            "gt_answer_snippet": "1. Press the red button. 2. Wait for the green light. 3. System will not automatically restart.",
            "gt_page_number": "-1",
            "parsed_steps": ["Press the red button.", "Wait for the green light.", "System will automatically restart."]
        }
    )

    base_candidate_path = Path(f"{output_base_path}{CANDIDATE_SUFFIX}")
    candidate_file_path_jsonl = base_candidate_path.with_suffix('.jsonl')
    try:
        if 'corrected_steps' not in df.columns:
            df['corrected_steps'] = None
        if 'procedural_comments' not in df.columns:
            df['procedural_comments'] = None
        with open(candidate_file_path_jsonl, 'w', encoding='utf-8') as f:
            df.to_json(f, orient='records', lines=True, force_ascii=False)
        logging.info(f"Final candidates saved in JSONL format to: {candidate_file_path_jsonl} ({len(df)} rows)")
    except Exception as e:
        logging.error(f"Failed to save candidate DataFrame to JSONL: {e}", exc_info=True)
        logging.warning("Continuing with audit file export, but candidate save failed.")

    print(f"\n--- Step 9: Exporting Audit & Review Files ---")

    print("\nAudit Export (Stage 1): Preparing General Quality audit files...")
    eligible_categories = [cat for cat in VALID_CATEGORIES if
                           cat not in [PROCEDURAL_CATEGORY_NAME, UNANSWERABLE_CATEGORY_NAME]]
    df_audit_pool_s1 = df[df['category'].isin(eligible_categories)]
    population_size = len(df_audit_pool_s1)

    if population_size > 0:
        num_to_sample = math.ceil(population_size * audit_fraction)
    else:
        num_to_sample = 0
    num_to_sample = min(num_to_sample, population_size)

    sentinel_df_s1 = pd.DataFrame([SENTINEL_ROWS_INFO[0][0]])
    if num_to_sample > 0:
        real_sample_df = df_audit_pool_s1.sample(n=num_to_sample, random_state=random_seed)
        stage1_audit_df = pd.concat([real_sample_df, sentinel_df_s1], ignore_index=True)
    else:
        stage1_audit_df = sentinel_df_s1
        logging.warning(f"Audit Export (Stage 1): Not enough eligible rows to sample. Exporting sentinel only.")

    stage1_audit_df = stage1_audit_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    core_cols_s1 = ["question_id", "question_text", "gt_answer_snippet", "gt_page_number"]
    label_cols_s1 = ['answer_correct?', 'grounded?', 'question_clear?']
    base_audit_df_s1 = stage1_audit_df.reindex(columns=core_cols_s1)
    print(f"  -> Stage 1 files will contain {len(base_audit_df_s1)} rows ({num_to_sample} real + 1 sentinel).")
    for rater in ["A", "B"]:
        audit_df = base_audit_df_s1.copy()
        for label_col in label_cols_s1:
            audit_df[label_col] = None
        suffix = f"_general_audit_{rater}.jsonl"
        audit_file_path = Path(f"{output_base_path.parent / output_base_path.stem}{suffix}")
        try:
            with open(audit_file_path, 'w', encoding='utf-8') as f:
                audit_df.to_json(f, orient='records', lines=True, force_ascii=False)
            print(f"  -> Stage 1 Audit file saved: {audit_file_path}")
        except Exception as e:
            logging.error(f"Failed to save audit file {audit_file_path}: {e}", exc_info=True)

    print("\nAudit Export (Stage 2): Preparing Procedural Quality audit files...")
    df_procedural = df[df['category'] == PROCEDURAL_CATEGORY_NAME].copy()
    num_procedural = len(df_procedural)
    if num_procedural > 0:
        sentinel_df_s2 = pd.DataFrame([SENTINEL_PROCEDURAL_INFO])
        stage2_audit_df = pd.concat([df_procedural, sentinel_df_s2], ignore_index=True)
        stage2_audit_df = stage2_audit_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        core_cols_s2 = ["question_id", "question_text", "gt_answer_snippet", "gt_page_number", "parsed_steps"]
        label_cols_s2 = ['answer_correct?', 'question_clear?', 'correct_number_of_steps?', 'correct_order_of_steps?']
        base_audit_df_s2 = stage2_audit_df.reindex(columns=core_cols_s2)
        print(f"  -> Stage 2 files will contain {len(base_audit_df_s2)} rows ({num_procedural} real + 1 sentinel).")
        for rater in ["A", "B"]:
            audit_df = base_audit_df_s2.copy()
            for label_col in label_cols_s2:
                audit_df[label_col] = None
            suffix = f"_procedural_audit_{rater}.jsonl"
            audit_file_path = Path(f"{output_base_path.parent / output_base_path.stem}{suffix}")
            try:
                with open(audit_file_path, 'w', encoding='utf-8') as f:
                    audit_df.to_json(f, orient='records', lines=True, force_ascii=False)
                print(f"  -> Stage 2 Audit file saved: {audit_file_path}")
            except Exception as e:
                logging.error(f"Failed to save audit file {audit_file_path}: {e}", exc_info=True)
    else:
        print("  -> No procedural rows found to export for Stage 2 audit.")

    print("\nAudit Export (Stage 3): Preparing procedural correction file (CSV format)...")
    review_file_path = Path(f"{output_base_path}_procedural_review.csv")
    df_procedural_for_review = df[df['category'] == PROCEDURAL_CATEGORY_NAME].copy()
    if not df_procedural_for_review.empty:
        review_cols = ["question_id", "question_text", "gt_answer_snippet", "gt_page_number"]
        if 'parsed_steps' in df_procedural_for_review.columns:
            review_cols.append('parsed_steps')
        procedural_review_df = df_procedural_for_review[review_cols].reset_index(drop=True)
        procedural_review_df['corrected_steps'] = ""
        procedural_review_df['procedural_comments'] = ""
        try:
            if 'parsed_steps' in procedural_review_df.columns:
                procedural_review_df['parsed_steps'] = procedural_review_df['parsed_steps'].apply(
                    lambda x: str(x) if x is not None else '[]')
            procedural_review_df.to_csv(review_file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
            print(f"  -> Stage 3 Correction file saved: {review_file_path}")
        except Exception as e:
            logging.error(f"Failed to save procedural review file {review_file_path}: {e}", exc_info=True)
    else:
        print("  -> No procedural rows found to export for Stage 3 correction.")

    print("\n--- Step 9 Complete (Audit/Review File Export) ---")
    return num_to_sample, num_procedural


# --- Main execution block ---
def main():
    parser = argparse.ArgumentParser(description="Generate & process QA dataset drafts.")
    parser.add_argument("-i", "--input", dest="input_jsonl", required=True,
                        help="Path to input JSONL file.")

    args = parser.parse_args()

    input_file_path = Path(args.input_jsonl)
    if not input_file_path.is_file():
        logging.error(f"Input file not found: {input_file_path}")
        sys.exit(1)
    output_base_path = input_file_path.parent / input_file_path.stem.replace("_pages", "")
    try:
        output_base_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create output dir: {e}")
        sys.exit(1)

    raw_output_file_path = Path(f"{output_base_path}{RAW_OUTPUT_SUFFIX}")
    gold_file_path = Path(f"{output_base_path}{GOLD_DATASET_SUFFIX}")
    stats_file_path = Path(f"{output_base_path}{STATS_SUFFIX}")
    procedural_review_file_path = Path(f"{output_base_path}_procedural_review.csv")

    print(f"--- Running in GENERATION mode for input: {args.input_jsonl} ---")
    print(f"--- All output files will be saved relative to: {output_base_path} ---")
    df = pd.DataFrame()
    loaded_pages = []
    loaded_doc_id = ""
    loaded_language = ""

    serializable_config = {}
    for k, v in config.items():
        try:
            json.dumps({k: v})
        except TypeError:
            serializable_config[k] = str(v)
        else:
            serializable_config[k] = v
    run_stats = {'manual_id': output_base_path.name, 'config_used': serializable_config}

    try:
        # Step 1: Load
        print("\n--- Step 1: Loading & Setup ---")
        loaded_pages, loaded_doc_id, loaded_language = load_manual(args.input_jsonl)
        run_stats['input_pages'] = len(loaded_pages)
        logging.info(f"Loaded {len(loaded_pages)} pages.")

        # Step 2: Get Raw Rows
        print("\n--- Step 2: Obtaining Raw Generated Rows ---")

        logging.info("Building prompt...")
        full_prompt = build_prompt(loaded_pages)
        logging.info(f"Prompt length: {len(full_prompt)} chars.")

        try:
            logging.info("Calculating prompt token count via API...")
            prompt_token_count = generation_client.count_tokens(full_prompt)
            run_stats['prompt_token_count'] = prompt_token_count
            logging.info(f"Prompt token count: {prompt_token_count}")
        except Exception as e:
            logging.warning(f"Could not calculate prompt token count: {e}")
            run_stats['prompt_token_count'] = -1

        loaded_raw_rows = []
        if raw_output_file_path.is_file():
            logging.info(f"Raw file found: {raw_output_file_path}. Skipping generation and loading rows from file...")
            raw_text = raw_output_file_path.read_text(encoding='utf-8')
            loaded_raw_rows = [l for l in raw_text.splitlines() if l.strip()]
            logging.info(f"Loaded {len(loaded_raw_rows)} lines.")
        else:
            logging.info(f"Raw file not found. Generating new rows...")
            raw_generated_rows = over_generate(full_prompt, OVERGEN_FACTOR, generation_client)
            logging.info(f"Target raw output file: {raw_output_file_path}")
            if raw_generated_rows:
                raw_output_content = "\n".join(raw_generated_rows)
                raw_output_file_path.write_text(raw_output_content, encoding='utf-8')
                logging.info(f"Saved {len(raw_generated_rows)} rows to {raw_output_file_path}")
                loaded_raw_rows = raw_generated_rows
            else:
                raise ValueError("Generation failed to produce any rows.")

        if not loaded_raw_rows:
            raise ValueError("No raw rows obtained (loaded or generated).")
        run_stats['raw_rows_obtained'] = len(loaded_raw_rows)

        # Step 3: Parse
        print("\n--- Step 3: Parsing Raw Rows ---")
        df = parse_rows(loaded_raw_rows, SCHEMA, loaded_doc_id, loaded_language)
        run_stats['rows_parsed'] = len(df)
        run_stats['parse_failures'] = run_stats.get('raw_rows_obtained', 0) - run_stats['rows_parsed']
        if df.empty:
            raise ValueError("Parsing failed, no valid rows produced.")
        print(f"\n--- Step 3 Verification ---")
        print(f"Parsed {df.shape[0]} rows.")
        if 'category' in df.columns:
            print("Category Distribution (Initial):\n", df['category'].value_counts().to_string())
        else:
            print("Category column missing after parse.")
        print("\n--- Step 3 Complete (Parsing) ---")

        # Step 3a: Filter Invalid Metadata
        rows_before_meta_filter = len(df)
        df = filter_invalid_metadata(df)
        run_stats['rows_after_meta_filter'] = len(df)
        run_stats['rows_removed_invalid_metadata'] = rows_before_meta_filter - len(df)
        if df.empty:
            raise ValueError("DataFrame empty after filtering invalid metadata.")
        print(f" Shape after Metadata Filter: {df.shape}")
        if 'category' in df.columns:
            print("Category Distribution (After Meta Filter):\n", df['category'].value_counts().to_string())

        # Step 3b: Add Parsed Steps via LLM
        df = add_parsed_steps_llm(df, step_parsing_client)
        proc_found = int((df['category'] == PROCEDURAL_CATEGORY_NAME).sum())
        proc_parsed = int(df['parsed_steps'].apply(
            lambda x: isinstance(x, list) and len(x) > 0).sum()) if 'parsed_steps' in df.columns else 0
        run_stats['procedural_rows_found'] = proc_found
        run_stats['procedural_rows_llm_parsed'] = proc_parsed
        print(f"\n--- Step 3b Verification ---")
        print(f"LLM parsed steps for {proc_parsed}/{proc_found} procedural rows.")

        # Step 4: Deduplicate
        rows_before = len(df)
        if not df.empty:
            df, similarity_scores = deduplicate(df, embedding_model, DUP_THRESHOLD, EMBED_MODEL)
            run_stats['deduplication_similarity_scores'] = similarity_scores
        else:
            run_stats['deduplication_similarity_scores'] = []

        run_stats['rows_after_dedupe'] = len(df)
        run_stats['rows_removed_dedupe'] = rows_before - len(df)
        if df.empty:
            logging.warning("DataFrame empty after Deduplication.")

        # Step 5: Annotate Page Check
        if not df.empty:
            df = page_check_annotate(df, loaded_pages)
        if not df.empty and 'passed_strict_check' in df.columns:
            passed_strict_count = int(df['passed_strict_check'].sum())
            failed_strict_count = len(df) - passed_strict_count
            run_stats['stats_after_annotate'] = {'total_rows': len(df), 'passed_strict': passed_strict_count,
                                                 'failed_strict': failed_strict_count}
            try:
                pass_rate_by_cat = df.groupby('category')['passed_strict_check'].mean().round(3).to_dict()
                run_stats['strict_check_rate_by_category'] = {k: v for k, v in pass_rate_by_cat.items() if pd.notna(v)}
            except Exception as e:
                logging.warning(f"Could not calc per-cat strict pass rate: {e}")
                run_stats['strict_check_rate_by_category'] = {}
        else:
            run_stats['stats_after_annotate'] = {'total_rows': len(df), 'passed_strict': 0, 'failed_strict': len(df)}
            run_stats['strict_check_rate_by_category'] = {}
        if df.empty:
            logging.warning("DataFrame empty after Page Check annotation.")

        # -----------------------------------------------------------------
        # --- NEW ABLATION PROXY IMPLEMENTATION (Replaces old Step 6 & 7) ---
        # -----------------------------------------------------------------

        # Define path for the ablation log
        ABLATION_LOG_SUFFIX = '_ablation_log.jsonl'
        ablation_log_path = Path(f"{output_base_path}{ABLATION_LOG_SUFFIX}")

        # Step 6: RAGAS *Scoring*
        rows_before_scoring = len(df)
        if not df.empty:
            # Call the new function (note: no threshold)
            df, faithfulness_scores, correctness_scores = add_ragas_scores(df, loaded_pages)
            run_stats['ragas_faithfulness_scores'] = faithfulness_scores
            run_stats['ragas_correctness_scores'] = correctness_scores
        else:
            run_stats['ragas_faithfulness_scores'] = []
            run_stats['ragas_correctness_scores'] = []

        # Step 7: Judge *Scoring*
        if not df.empty:
            # Call the new function (note: no threshold)
            # This adds 'judge_score' column to the *same* df
            df, judge_scores = add_judge_scores(df, loaded_pages, judge_client)
            run_stats['judge_scores'] = judge_scores
        else:
            run_stats['judge_scores'] = []

        # --- ABLATION PROXY: SAVE SCORES ---
        # At this point, 'df' contains ALL candidates from step 5,
        # plus scores from BOTH RAGAS and the Judge.
        if not df.empty:
            try:
                logging.info(f"Saving ablation proxy data ({len(df)} rows) to: {ablation_log_path}")
                # Save the full dataframe with ALL scores before filtering
                df.to_json(ablation_log_path, orient='records', lines=True, force_ascii=False)
                run_stats['ablation_log_saved'] = True
                print(f"*** Ablation log saved to {ablation_log_path} ***")
            except Exception as e:
                logging.error(f"Failed to save ablation log: {e}", exc_info=True)
                run_stats['ablation_log_saved'] = False

        # --- NOW, APPLY FILTERS SEQUENTIALLY ---
        logging.info("Applying RAGAS and Judge filters sequentially to scored DataFrame...")

        # Apply RAGAS Filter (Step 6 Filter)
        rows_before_ragas_filter = len(df)
        if not df.empty and 'ragas_faithfulness' in df.columns:
            unanswerable_mask = df['category'] == UNANSWERABLE_CATEGORY_NAME
            passed_ragas_mask = (df['ragas_faithfulness'] >= RAGAS_THRESHOLD) & (
                        df['ragas_correctness'] >= RAGAS_THRESHOLD)

            df = df[unanswerable_mask | passed_ragas_mask].copy()  # .copy() to avoid SettingWithCopyWarning
            logging.info(f"Rows remaining after RAGAS filter: {len(df)}")

        run_stats['rows_after_ragas'] = len(df)
        run_stats['rows_removed_ragas'] = rows_before_ragas_filter - len(df)
        if df.empty:
            logging.warning("DataFrame empty after Ragas filtering.")

        # Apply Judge Filter (Step 7 Filter)
        rows_before_judge_filter = len(df)
        if not df.empty and 'judge_score' in df.columns:
            unanswerable_mask = df['category'] == UNANSWERABLE_CATEGORY_NAME
            passed_judge_mask = (df['judge_score'] >= JUDGE_THRESHOLD)

            # Rows must be (Unanswerable) OR (Pass Judge)
            # RAGAS failures are already gone, but this logic is robust
            df = df[unanswerable_mask | passed_judge_mask].reset_index(drop=True)
            logging.info(f"Rows remaining after Judge filter: {len(df)}")

        run_stats['rows_after_judge'] = len(df)
        run_stats['rows_removed_judge'] = rows_before_judge_filter - len(df)
        if df.empty:
            logging.warning("DataFrame empty after LLM Judge filtering.")

        # Step 7b: Verify Unanswerable
        rows_before = len(df)
        unans_before = (df['category'] == UNANSWERABLE_CATEGORY_NAME).sum() if 'category' in df.columns else 0
        run_stats['unanswerable_initially'] = int(unans_before)
        if not df.empty:
            df_ans = df[df['category'] != UNANSWERABLE_CATEGORY_NAME].copy()
            df_unans = df[df['category'] == UNANSWERABLE_CATEGORY_NAME].copy()
            if not df_unans.empty:
                full_ctx = get_full_manual_text(loaded_pages)
                verified_idx = verify_unanswerable(df_unans, full_ctx, judge_client)
                df_veri_unans = df_unans.loc[verified_idx]
                df = pd.concat([df_ans, df_veri_unans]).sort_index().reset_index(drop=True)
            else:
                logging.info("No Unanswerable rows to verify.")
        unans_after = (df['category'] == UNANSWERABLE_CATEGORY_NAME).sum() if 'category' in df.columns else 0
        run_stats['unanswerable_verified'] = int(unans_after)
        run_stats['rows_after_unanswerable_verify'] = len(df)
        print(f" Shape after Unanswerable check: {df.shape}")

        # Step 8: Quota Select
        if df.empty:
            raise ValueError("No rows left for quota selection.")
        df, quota_info = quota_select(df, CATEGORY_TARGETS, FINAL_DATASET_SIZE)
        run_stats['quota_selection_details'] = quota_info
        run_stats['quota_met_fully'] = quota_info.get('_summary', {}).get('all_quotas_met', False)
        run_stats['final_dataset_size_generated'] = len(df)
        if len(df) == 0:
            raise ValueError("Quota selection resulted in zero rows. Cannot proceed.")

        # Step 9: Save Candidates & Export Audit Slice
        if df.empty or len(df) < 1:
            raise ValueError(f"Quota selection failed or resulted in empty DataFrame. Cannot export.")

        general_audit_size, procedural_audit_size = export_audit_slice(df, output_base_path, AUDIT_FRACTION)

        run_stats['audit_sample_size_stage1'] = general_audit_size + NUM_SENTINELS
        run_stats['audit_sample_size_stage2'] = procedural_audit_size + (1 if procedural_audit_size > 0 else 0)

        # Save Stats for Generate Mode
        run_stats['pipeline_status'] = 'Generated'
        try:
            def make_serializable(obj):
                if isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                if isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                if isinstance(obj, dict):
                    return {str(k): make_serializable(v) for k, v in obj.items()}
                if hasattr(obj, 'item'):
                    return obj.item()
                if isinstance(obj, Path):
                    return str(obj)
                return str(obj)


            serializable_stats = make_serializable(run_stats)
            with open(stats_file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2)
            logging.info(f"Run statistics saved to: {stats_file_path}")
        except Exception as e:
            logging.error(f"Failed to save run statistics: {e}", exc_info=True)

        # Final Instructions
        print("\n*** GENERATION COMPLETE ***")
        print("Candidate data and run statistics have been saved.")

        base_name = output_base_path.stem
        general_audit_file_A = f"{base_name}_general_audit_A.jsonl"
        procedural_audit_file_A = f"{base_name}_procedural_audit_A.jsonl"
        procedural_correction_file = procedural_review_file_path.name

        print(f"\nThree sets of files have been exported for review:")
        print(f"  1. General Quality Audit: '{general_audit_file_A}' (and B)")
        print(f"  2. Procedural Quality Audit: '{procedural_audit_file_A}' (and B)")
        print(f"  3. Procedural Step Correction: '{procedural_correction_file}'")

        print("\nACTION REQUIRED:")
        print("1. Perform the two-stage audit using the JSONL files and your rater instructions.")
        print(f"2. (Optional) For subject matter experts: correct the procedural steps by editing the")
        print(f"   'corrected_steps' and 'procedural_comments' columns in the '{procedural_correction_file}' CSV file.")

    except Exception as e:
        logging.error(f"An error occurred during the generation pipeline: {e}", exc_info=True)
        try:
            if 'run_stats' in locals() and run_stats:
                run_stats['pipeline_status'] = 'Failed'
                run_stats['error_message'] = str(e)
                serializable_stats = {}
                for k, v in run_stats.items():
                    try:
                        json.dumps({k: v})
                    except TypeError:
                        serializable_stats[k] = str(v)
                    else:
                        serializable_stats[k] = v
                with open(stats_file_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_stats, f, indent=2)
                logging.info(f"Partial run statistics saved to {stats_file_path} due to error.")
        except Exception as dump_e:
            logging.error(f"Could not save partial stats on error: {dump_e}")
        sys.exit(1)

    print("\nScript finished generate mode.")


if __name__ == "__main__":
    main()